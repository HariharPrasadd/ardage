## Advances and Challenges in Meta-Learning: A Technical Review

Anna Vettoruzzo [1], Mohamed-Rafik Bouguelia [1], Joaquin
Vanschoren [2], Thorsteinn R¨ognvaldsson [1], and KC Santosh [3]


1
_Center for Applied Intelligent Systems Research (CAISR), Halmstad University, Sweden_


2 _Automated Machine Learning Group, Eindhoven University of Technology, Netherlands_


3 _Applied AI Research Lab, Computer Science, University of South Dakota, USA_


**Abstract**


Meta-learning empowers learning systems with the ability to acquire
knowledge from multiple tasks, enabling faster adaptation and generalization to new tasks. This review provides a comprehensive technical overview
of meta-learning, emphasizing its importance in real-world applications
where data may be scarce or expensive to obtain. The paper covers the stateof-the-art meta-learning approaches and explores the relationship between
meta-learning and multi-task learning, transfer learning, domain adaptation and generalization, self-supervised learning, personalized federated
learning, and continual learning. By highlighting the synergies between
these topics and the field of meta-learning, the paper demonstrates how
advancements in one area can benefit the field as a whole, while avoiding unnecessary duplication of efforts. Additionally, the paper delves into
advanced meta-learning topics such as learning from complex multi-modal
task distributions, unsupervised meta-learning, learning to efficiently adapt
to data distribution shifts, and continual meta-learning. Lastly, the paper
highlights open problems and challenges for future research in the field. By
synthesizing the latest research developments, this paper provides a thorough understanding of meta-learning and its potential impact on various
machine learning applications. We believe that this technical overview will
contribute to the advancement of meta-learning and its practical implications in addressing real-world problems.


**Keywords** : Meta-learning, transfer learning, few-shot learning, representation learning, deep neural networks

### **1 Introduction**


**Context and motivation**


Deep representation learning has revolutionized the field of machine learning
by enabling models to learn effective features from data. However, it often requires large amounts of data for solving a specific task, making it impractical in


1


©This work has been submitted to the IEEE for possible publication. Copyright may be transferred
without notice, after which this version may no longer be accessible.


scenarios where data is scarce or costly to obtain. Most existing approaches rely
on either supervised learning of a representation tailored to a single task, or unsupervised learning of a representation that captures general features that may
not be well-suited to new tasks. Furthermore, learning from scratch for each
task is often not feasible, especially in domains such as medicine, robotics, and
rare language translation where data availability is limited.
To overcome these challenges, meta-learning has emerged as a promising
approach. Meta-learning enables models to quickly adapt to new tasks, even
with few examples, and generalize across them. While meta-learning shares
similarities with transfer learning and multitask learning, it goes beyond these
approaches by enabling a learning system to _learn how to learn_ . This capability is
particularly valuable in settings where data is scarce, costly to obtain, or where
the environment is constantly changing. While humans can rapidly acquire
new skills by leveraging prior experience and are therefore considered _general-_
_ists_, most deep learning models are still _specialists_ and are limited to performing
well on specific tasks. Meta-learning bridges this gap by enabling models to efficiently adapt to new tasks.


**Contribution**


This review paper primarily discusses the use of meta-learning techniques in
deep neural networks to learn reusable representations, with an emphasis on
few-shot learning; it does not cover topics such as AutoML and Neural Architecture Search [1], which are out of scope. Distinct from existing surveys on
meta-learning, such as [2, 3, 4, 5], this review paper highlights several key differentiating factors:


  - **Inclusion of advanced meta-learning topics** . In addition to covering fundamental aspects of meta-learning, this review paper delves into advanced
topics such as learning from multimodal task distributions, meta-learning
without explicit task information, learning without data sharing among
clients, adapting to distribution shifts, and continual learning from a stream
of tasks. By including these advanced topics, our paper provides a comprehensive understanding of the current state-of-the-art and highlights
the challenges and opportunities in these areas.


  - **Detailed exploration of relationship with other topics** . We not only examine meta-learning techniques but also establish clear connections between meta-learning and related areas, including transfer learning, multitask learning, self-supervised learning, personalized federated learning,
and continual learning. This exploration of the relationships and synergies between meta-learning and these important topics provides valuable
insights into how meta-learning can be efficiently integrated into broader
machine learning frameworks.


  - **Clear and concise exposition** . Recognizing the complexity of meta-learning,
this review paper provides a clear and concise explanation of the concepts, techniques and applications of meta-learning. It is written with the
intention of being accessible to a wide range of readers, including both
researchers and practitioners. Through intuitive explanations, illustrative


2


examples, and references to seminal works, we facilitate readers’ understanding of the foundation of meta-learning and its practical implications.


  - **Consolidation of key information** . As a fast-growing field, meta-learning
has information scattered across various sources. This review paper consolidates the most important and relevant information about meta-learning,
presenting a comprehensive overview in a single resource. By synthesizing the latest research developments, this survey becomes an indispensable guide to researchers and practitioners seeking a thorough understanding of meta-learning and its potential impact on various machine
learning applications.


By highlighting these contributions, this paper complements existing surveys
and offers unique insights into the current state and future directions of metalearning.


**Organization**


In this paper, we provide the foundations of modern deep learning methods
for learning across tasks. To do so, we first define the key concepts and introduce relevant notations used throughout the paper in section 2. Then, we cover
the basics of multitask learning and transfer learning and their relation to metalearning in section 3. In section 4, we present an overview of the current state of
meta-learning methods and provide a unified view that allows us to categorize
them into three types: black-box meta-learning methods, optimization-based
meta-learning methods, and meta-learning methods that are based on distance
metric learning [6]. In section 5, we delve into advanced meta-learning topics, explaining the relationship between meta-learning and other important machine learning topics, and addressing issues such as learning from multimodal
task distributions, performing meta-learning without provided tasks, learning
without sharing data across clients, learning to adapt to distribution shifts, and
continual learning from a stream of tasks. Finally, the paper explores the application of meta-learning to real-world problems and provides an overview of
the landscape of promising frontiers and yet-to-be-conquered challenges that
lie ahead. Section 6 focuses on these challenges, shedding light on the most
pressing questions and future research opportunities.

### **2 Basic notations and definitions**


In this section, we introduce some simple notations which will be used throughout the paper and provide a formal definition of the term “task” within the
scope of this paper.
We use _θ_ (and sometimes also _ϕ_ ) to represent the set of parameters (weights)
of a deep neural network model. _D_ = _{_ ( _x_ _j_ _, y_ _j_ ) _}_ _[n]_ _j_ =1 [denotes a dataset, where]
inputs _x_ _j_ are sampled from the distribution _p_ ( _x_ ) and outputs _y_ _j_ are sampled
from _p_ ( _y|x_ ). The function _L_ ( _., ._ ) denotes a loss function, for example, _L_ ( _θ, D_ )
represents the loss achieved by the model’s parameters _θ_ on the dataset _D_ . The
symbol _T_ refers to a task, which is primarily defined by the data-generating
distributions _p_ ( _x_ ) and _p_ ( _y|x_ ) that define the problem.


3


In a standard supervised learning scenario, the objective is to optimize the
parameters _θ_ by minimizing the loss _L_ ( _θ, D_ ), where the dataset _D_ is derived
from a single task _T_, and the loss function _L_ depends on that task. Formally,
in this setting, a task _T_ _i_ is a triplet _T_ _i_ ≜ _{p_ _i_ ( _x_ ) _, p_ _i_ ( _y|x_ ) _, L_ _i_ _}_ that includes taskspecific data-generating distributions _p_ _i_ ( _x_ ) and _p_ _i_ ( _y|x_ ), as well as a task-specific
loss function _L_ _i_ . The goal is to learn a model that performs well on data sampled from task _T_ _i_ . In a more challenging setting, we consider learning from
multiple tasks _{T_ _i_ _}_ _[T]_ _i_ =1 [, which involves (a dataset of) multiple datasets] _[ {D]_ _[i]_ _[}]_ _[T]_ _i_ =1 [.]
In this scenario, a set of training tasks is used to learn a model that performs
well on test tasks. Depending on the specific setting, a test task can either be
sampled from the training tasks or completely new, never encountered during
the training phase.
In general, tasks can differ in various ways depending on the application.
For example, in image recognition, different tasks can involve recognizing handwritten digits or alphabets from different languages [7, 8], while in natural language processing, tasks can include sentiment analysis [9, 10], machine translation [11], and chatbot response generation [12, 13, 14]. Tasks in robotics can
involve training robots to achieve different goals [15], while in automated feedback generation, tasks can include providing feedback to students on different
exams [16]. It is worth noting that tasks can share structures, even if they appear unrelated. For example, the laws of physics underlying real data, the language rules underlying text data, and the intentions of people all share common
structures that enable models to transfer knowledge across seemingly unrelated
tasks.

### **3 From multitask and transfer to meta-learning**


Meta-learning, multitask learning, and transfer learning encompass different
approaches aimed at learning across multiple tasks. Multitask learning aims to
improve performance on a set of tasks by learning them simultaneously. Transfer learning fine-tunes a pre-trained model on a new task with limited data. In
contrast, meta-learning acquires useful knowledge from past tasks and leverages it to learn new tasks more efficiently. In this section, we transition from
discussing “multitask learning” and “transfer learning” to introducing the topic
of “meta-learning”.


**3.1** **Multitask learning problem**


As illustrated in Figure 1 (A), multitask learning (MTL) trains a model to perform multiple related tasks simultaneously, leveraging shared structure across
tasks, and improving performance compared to learning each task individually. In this setting, there is no distinction between training and test tasks, and
we refer to them as _{T_ _i_ _}_ _[T]_ _i_ =1 [.]
One common approach in MTL is hard parameter sharing, where the model
parameters _θ_ are split into shared _θ_ [sh] and task-specific _θ_ _[i]_ parameters. These
parameters are learned simultaneously through an objective function that takes
the form:



min
_θ_ [sh] _,θ_ [1] _,...,θ_ _[T]_



_T_
� _w_ _i_ _L_ _i_ ( _{θ_ [sh] _, θ_ _[i]_ _}, D_ _i_ ) _,_


_i_ =1


4


Figure 1: Multitask learning vs transfer learning vs meta-learning.


where _w_ _i_ can weight tasks differently. This approach is often implemented using a multi-headed neural network architecture, where a shared encoder (parameterized by _θ_ [sh] ) is responsible for feature extraction. This shared encoder
subsequently branches out into task-specific decoding heads (parameterized by
_θ_ _[i]_ ) dedicated to individual tasks _T_ _i_ [17, 18, 19].
Soft parameter sharing is another approach in MTL that encourages parameter similarity across task-specific models using regularization penalties [20,
21, 22]. In this approach, each task typically has its own model with its own set
of parameters _θ_ _[i]_, while the shared parameters set _θ_ [sh] can be empty. The objective function is similar to that of hard parameter sharing, but with an additional
regularization term that controls the strength of parameter sharing across tasks.
The strength of regularization is determined by the hyperparameter _λ_ . In the
case of _L_ 2 regularization, the objective function is given by:



_T_
� _∥θ_ _[i]_ _−_ _θ_ _[i]_ _[′]_ _∥._


_i_ _[′]_ =1



min
_θ_ [sh] _,θ_ [1] _,...,θ_ _[T]_



_T_
� _w_ _i_ _L_ _i_ ( _{θ_ [sh] _, θ_ _[i]_ _}, D_ _i_ ) + _λ_


_i_ =1



However, soft parameter sharing can be more memory-intensive as separate
sets of parameters are stored for each task, and it requires additional design
decisions and hyperparameters.
Another approach to sharing parameters is to condition a single model on a
task descriptor _z_ _i_ that contains task-specific information used to modulate the
network’s computation. The task descriptor _z_ _i_ can be a simple one-hot encoding of the task index or a more complex task specification, such as language


5


description or user attributes. When a task descriptor is provided, it is used to
modulate the weights of the shared network with respect to the task at hand.
Through this modulation mechanism, the significance of the shared features is
determined based on the particular task, enabling the learning of both shared
and task-specific features in a flexible manner. Such an approach grants finegrained control over the adjustment of the network’s representation, tailoring
it to each individual task. Various methods for conditioning the model on the
task descriptor are described in [23]. More complex methods are also provided
in [24, 25, 26].
Choosing the appropriate approach for parameter sharing, determining the
level of the network architecture at which to share parameters, and deciding
on the degree of parameter sharing across tasks are all design decisions that
depend on the problem at hand. Currently, these decisions rely on intuition
and knowledge of the problem, making them more of an art than a science,
similar to the process of tuning neural network architectures. Moreover, multitask learning presents several challenges, such as determining which tasks
are complementary, particularly in scenarios with a large number of tasks, as in

[27]. Interested readers can find a more comprehensive discussion of multitask
learning in [28, 29].
In summary, multitask learning aims to learn a set of _T_ tasks _{T_ _i_ _}_ _[T]_ _i_ =1 [at once.]
Even though the model can generalize to new data from these _T_ tasks, it might
not be able to handle a completely new task that it has not been trained on. This
is where transfer learning and meta-learning become more relevant.


**3.2** **Transfer learning via fine-tuning**


Transfer learning is a valuable technique that allows a model to leverage representations learned from one or more source tasks to solve a target task. As
illustrated in Figure 1 (B), the main goal is to use the knowledge learned from
the source task(s) _T_ _a_ to improve the performance of the model on a new task,
usually referred to as the target task _T_ _b_, especially when the target task dataset
_D_ _b_ is limited. In practice, the source task data _D_ _a_ is often inaccessible, either
because it is too expensive to obtain or too large to store.
One common approach for transfer learning is fine-tuning, which involves
starting with a model that has been pre-trained on the source task dataset _D_ _a_ .
The parameters of the pre-trained model, denoted as _θ_, are then fine-tuned on
the training data _D_ _b_ from the target task _T_ _b_ using gradient descent or any other
optimizer for several optimization steps. An example of the fine-tuning process
for one gradient descent step is expressed as follows:


_ϕ ←_ _θ −_ _α∇_ _θ_ _L_ ( _θ, D_ _b_ ) _,_


where _ϕ_ denotes the parameters fine-tuned for task _T_ _b_, and _α_ is the learning
rate.
Models with pre-trained parameters _θ_ are often available online, including
models pre-trained on large datasets such as ImageNet for image classification

[30] and language models like BERT [31], PaLM [32], LLaMA [33], and GPT-4

[34], trained on large text corpora. Models pre-trained on other large and diverse datasets or using unsupervised learning techniques, as discussed in section 5.3, can also be used as a starting point for fine-tuning.


6


However, as discussed in [35], it is crucial to avoid destroying initialized
features when fine-tuning. Some design choices, such as using a smaller learning rate for earlier layers, freezing earlier layers and gradually unfreezing, or
re-initializing the last layer, can help to prevent this issue. Recent studies such
as [36] show that fine-tuning the first or middle layers can sometimes work
better than fine-tuning the last layers, while others recommend a two-step process of training the last layer first and then fine-tuning the entire network [35].
More advanced approaches, such as STILTs [37], propose an intermediate step
of further training the model on a labeled task with abundant data to mitigate
the potential degradation of pre-trained features.
In [38], it was demonstrated that transfer learning via fine-tuning may not
always be effective, particularly when the target task dataset is very small or
very different from the source tasks. To investigate this, the authors fine-tuned
a pre-trained universal language model on specific text corpora corresponding
to new tasks using varying numbers of training examples. Their results showed
that starting with a pre-trained model outperformed training from scratch on
the new task. However, when the size of the new task dataset was very small,
fine-tuning on such a limited number of examples led to poor generalization
performance. To address this issue, meta-learning can be used to learn a model
that can effectively adapt to new tasks with limited data by leveraging prior
knowledge from other tasks. In fact, meta-learning is particularly useful for
learning new tasks from very few examples, and we will discuss it in more
detail in the remainder of this paper.


**3.3** **Meta-learning problem**


Meta-learning (or learning to learn) is a field that aims to surpass the limitations of traditional transfer learning by adopting a more sophisticated approach
that explicitly optimizes for transferability. As discussed in section 3.2, traditional transfer learning involves pre-training a model on source tasks and finetuning it for a new task. In contrast, meta-learning trains a network to efficiently
learn or adapt to new tasks with only a few examples. Figure 1 (C) illustrates
this approach, where at meta-training time we _learn to learn_ tasks, and at metatest time we _learn_ a new task efficiently.
During the meta-training phase, prior knowledge enabling efficient learning
of new tasks is extracted from a set of training tasks _{T_ _i_ _}_ _[T]_ _i_ =1 [. This is achieved by]
using a meta-dataset consisting of multiple datasets _{D_ _i_ _}_ _[T]_ _i_ =1 [, each correspond-]
ing to a different training task. At meta-test time, a small training dataset _D_ new
is observed from a completely new task _T_ new and used in conjunction with the
prior knowledge to infer the most likely posterior parameters. As in transfer
learning, accessing prior tasks at meta-test time is impractical. Although the
datasets _{D_ _i_ _}_ _i_ come from different data distributions (since they come from
different tasks _{T_ _i_ _}_ _i_ ), it is assumed that the tasks themselves (both for training
and testing) are drawn i.i.d. from an underlying task distribution _p_ ( _T_ ), implying some similarities in the task structure. This assumption ensures the effectiveness of meta-learning frameworks even when faced with limited labeled
data. Moreover, the more tasks that are available for meta-training, the better
the model can learn to adapt to new tasks, just as having more data improves
performance in traditional machine learning.


7


In the next section, we provide a more formal definition of meta-learning
and various approaches to it.

### **4 Meta-learning methods**


To gain a unified understanding of the meta-learning problem, we can draw
an analogy to the standard supervised learning setting. In the latter, the goal
is to learn a set of parameters _ϕ_ for a base model _h_ _ϕ_ (e.g., a neural network
parametrized by _ϕ_ ), which maps input data _x ∈X_ to the corresponding output
_y ∈Y_ as follows:
_h_ _ϕ_ : _X →Y_ (1)
_x �→_ _y_ = _h_ _ϕ_ ( _x_ ) _._


To accomplish this, a typically large training dataset _D_ = _{_ ( _x_ _j_ _, y_ _j_ ) _}_ _[n]_ _j_ =1 [specific]
to a particular task _T_ is used to learn _ϕ_ .
In the meta-learning setting, the objective is to learn prior knowledge, which
consists of a set of meta-parameters _θ_, for a procedure _F_ _θ_ ( _D_ _i_ [tr] _[, x]_ [ts] [)][. This pro-]
cedure uses _θ_ to efficiently learn from (or adapt to) a small training dataset
_D_ _i_ [tr] [=] _[ {]_ [(] _[x]_ _[k]_ _[, y]_ _[k]_ [)] _[}]_ _[K]_ _k_ =1 [from a task] _[ T]_ _[i]_ [, and then make accurate predictions on un-]
labeled test data _x_ [ts] from the same task _T_ _i_ . As we will see in the following
sections, _F_ _θ_ is typically composed of two functions: (1) a meta-learner _f_ _θ_ ( _._ )
that produces task-specific parameters _ϕ_ _i_ _∈_ Φ from _D_ _i_ [tr] _[∈X]_ _[ K]_ [, and (2) a base]
model _h_ _ϕ_ _i_ ( _._ ) that predicts outputs corresponding to the data in _x_ [ts] :


_f_ _θ_ : _X_ _[K]_ _→_ Φ _h_ _ϕ_ _i_ : _X →Y_ (2)
_D_ _i_ [tr] _[�→]_ _[ϕ]_ _[i]_ [ =] _[ f]_ _[θ]_ [(] _[D]_ _i_ [tr] [)] _[,]_ _x �→_ _y_ = _h_ _ϕ_ _i_ ( _x_ ) _._


Note that the process of obtaining task-specific parameters _ϕ_ _i_ = _f_ _θ_ ( _D_ _i_ [tr] [)][ is often]
referred to as “ _adaptation_ ” in the literature, as it adapts to the task _T_ _i_ using a
small amount of data while leveraging the prior knowledge summarized in _θ_ .
The objective of meta-training is to learn the set of meta-parameters _θ_ . This is
accomplished by using a meta-dataset _{D_ _i_ _}_ _[T]_ _i_ =1 [, which consists of a dataset of]
datasets, where each dataset _D_ _i_ = _{_ ( _x_ _j_ _, y_ _j_ ) _}_ _[n]_ _j_ =1 [is specific to a task] _[ T]_ _[i]_ [.]
The unified view of meta-learning presented here is beneficial because it
simplifies the meta-learning problem by reducing it to the design and optimization of _F_ _θ_ . Moreover, it facilitates the categorization of the various metalearning approaches into three categories: black-box meta-learning methods,
optimization-based meta-learning methods, and distance metric-based metalearning methods (as discussed in [6]). An overview of these categories is provided in the subsequent sections.


**4.1** **Black-box meta-learning methods**


Black-box meta-learning methods represent _f_ _θ_ as a black-box neural network
that takes the entire training dataset, _D_ _i_ [tr] [, and predicts task-specific-parameters,]
_ϕ_ _i_ . These parameters are then used to parameterize the base network, _h_ _ϕ_ _i_, and
make predictions for test data-points, _y_ [ts] = _h_ _ϕ_ _i_ ( _x_ [ts] ). The architecture of this
approach is shown in Figure 2. The meta-parameters, _θ_, are optimized as shown
in Equation 3, and a general algorithm for these kinds of black-box methods is


8


Figure 2: Black-box meta-learning.


**Algorithm 1** Black-box meta-learning

1: Randomly initialize _θ_
2: **while** not done **do**
3: Sample a task _T_ _i_ _∼_ _p_ ( _T_ ) _(or a mini-batch of tasks)_

4: Sample disjoint datasets _D_ _i_ [tr] _[,][ D]_ _i_ [ts] [from] _[ T]_ _[i]_
5: Compute _ϕ_ _i_ _←_ _f_ _θ_ ( _D_ _i_ [tr] [)]
6: Update _θ_ using _∇_ _θ_ _L_ ( _ϕ_ _i_ _, D_ _i_ [ts] [)]
7: **end while**

8: **return** _θ_


outlined in Algorithm 1.



min
_θ_



�



_L_ ( _f_ _θ_ ( _D_ _i_ [tr] [)]
��� ~~�~~
_T_ _i_
_ϕ_ _i_



_, D_ _i_ [ts] [)] _[.]_ (3)



However, this approach faces a major challenge: outputting all the parameters _ϕ_ _i_ of the base network _h_ _ϕ_ _i_ is not scalable and is impractical for largescale models. To overcome this issue, black-box meta-learning methods, such
as MANN [39] and SNAIL [40], only output sufficient statistics instead of the
complete set of parameters of the base network. These methods allow _f_ _θ_ to
output a low-dimensional vector _z_ _i_ that encodes contextual task information,
rather than a full set of parameters _ϕ_ _i_ . In this case, _ϕ_ _i_ consists of _{z_ _i_ _, θ_ _h_ _}_, where
_θ_ _h_ denotes the trainable parameters of the network _h_ _ϕ_ _i_ . The base network _h_ _ϕ_ _i_
is modulated with task descriptors by using various techniques for _conditioning_
_on task descriptors_ discussed in section 3.1.
Several black-box meta-learning methods adopt different neural network architectures to represent _f_ _θ_ . For instance, methods described in [39], use LSTMs
or architectures with augmented memory capacities, such as Neural Turing
Machines, while others, like Meta Networks [41], employ external memory
mechanisms. SNAIL [40] defines meta-learner architectures that leverage temporal convolutions to aggregate information from past experience and attention mechanisms to pinpoint specific pieces of information. Alternatively, some
methods, such as the one proposed in [42], use a feedforward plus averaging
strategy. This latter feeds each data-point in _D_ _i_ [tr] [=] _[ {]_ [(] _[x]_ _[j]_ _[, y]_ _[j]_ [)] _[}]_ _[K]_ _j_ =1 [through a neu-]
ral network to produce a representation _r_ _j_ for each data-point, and then averages these representations to create a task representation _z_ _i_ = _K_ [1] � _Kj_ =1 _[r]_ _[j]_ [. This]

strategy may be more effective than using a recurrent model such as LSTM, as it


9


Figure 3: Optimization-based meta-learning with gradient-based optimization.


does not rely on the assumption of temporal relationships between data-points
in _D_ _i_ [tr] [.]
Black-box meta-learning methods are expressive, versatile, and easy to combine with various learning problems, including classification, regression, and
reinforcement learning. However, they require complex architectures for the
meta-learner _f_ _θ_, making them computationally demanding and data-inefficient.
As an alternative, one can represent _ϕ_ _i_ = _f_ _θ_ ( _D_ _i_ [tr] [)][ as an optimization procedure]
instead of a neural network. The next section explores methods that utilize this
approach.


**4.2** **Optimization-based meta-learning methods**


Optimization-based meta-learning offers an alternative to the black-box approach,
where the meta-learner _f_ _θ_ is an optimization procedure like gradient descent,
rather than a black-box neural network. The goal of optimization-based metalearning is to acquire a set of meta-parameters _θ_ that are easy to learn via gradient descent and to fine-tune on new tasks. Most optimization-based techniques
do so by defining meta-learning as a bi-level optimization problem. At the inner level, _f_ _θ_ produces task-specific parameters _ϕ_ _i_ using _D_ _i_ [tr] [, while at the outer]
level, the initial set of meta-parameters _θ_ is updated by optimizing the performance of _h_ _ϕ_ _i_ on the test set of the same task. This is shown in Figure 3 and in
Algorithm 2 in case _f_ _θ_ is a gradient-based optimization. The meta-parameters _θ_
can represent inner optimizers [43, 44, 45, 46], neural network architectures [47,
48], other network hyperparameters [49], or the initialization of the base model
_h_ ( _._ ) [8]. The latter approach is similar to transfer learning via fine-tuning (cf.
section 3.2), but instead of using a pre-trained _θ_ that may not be transferable to
new tasks, we learn _θ_ to explicitly optimize for transferability.
Model-Agnostic Meta-Learning (MAML) [8] is one of the earliest and most
popular optimization-based meta-learning methods. The main idea behind
MAML is to learn a set of initial neural network’s parameters _θ_ that can easily
be fine-tuned for any task using gradient descent with only a few steps. During
the meta-training phase, MAML minimizes the objective defined as follows:



min
_θ_



�



_L_ ( _θ −_ _α∇_ _θ_ _L_ ( _θ, D_ _i_ [tr] [)]
_T_ _i_ ~~�~~ ~~�~~ � ~~�~~
_ϕ_ _i_



_, D_ _i_ [ts] [)] _[.]_ (4)



10


**Algorithm 2** Optimization-based meta-learning with gradient-based optimization

1: Randomly initialize _θ_
2: **while** not done **do**

3: Sample a task _T_ _i_ _∼_ _p_ ( _T_ ) _(or a mini-batch of tasks)_
4: Sample disjoint datasets _D_ _i_ [tr] _[,][ D]_ _i_ [ts] [from] _[ T]_ _[i]_
5: Optimize _ϕ_ _i_ _←_ _θ −_ _α∇_ _θ_ _L_ ( _θ, D_ _i_ [tr] [)]
6: Update _θ_ using _∇_ _θ_ _L_ ( _ϕ_ _i_ _, D_ _i_ [ts] [)]
7: **end while**


8: **return** _θ_


Note that in Equation 4, the task-specific parameters _ϕ_ _i_ are obtained through a
single gradient descent step from _θ_, although in practice, a few more gradient
steps are usually used for better performance.
As a result, MAML produces a model initialization _θ_ that can be quickly
adapted to new tasks with a small number of training examples. Algorithm
2 can be viewed as a simplified illustration of MAML, where _θ_ represents the
parameters of a neural network. This is similar to Algorithm 1 but with _ϕ_ _i_ obtained through optimization.
During meta-test time, a small dataset _D_ new [tr] [is observed from a new task]
_T_ new _∼_ _p_ ( _T_ ). The goal is to use the prior knowledge encoded in _θ_ to train a
model that generalizes well to new, unseen examples from this task. To achieve
this, _θ_ is fine-tuned with a few adaptation steps using _∇_ _θ_ _L_ ( _θ, D_ new [tr] [)][, resulting]
in task-specific parameters _ϕ_ . These parameters are then used to make accurate
predictions on previously unseen input data from _T_ new .
MAML can be thought of as a computation graph (as shown in Figure 4)
with an embedded gradient operator. Interestingly, the components of this
graph can be interchanged or replaced with components from the black-box
approach. For instance, [43] also learned an initialization _θ_, but adapted _θ_
differently by using a learned network _f_ _w_ ( _θ, D_ _i_ [tr] _[,][ ∇]_ _[θ]_ _[L]_ [)][ instead of the gradient]
_∇_ _θ_ _L_ ( _θ, D_ _i_ [tr] [)][:]
_ϕ_ _i_ _←_ _θ −_ _αf_ _w_ ( _θ, D_ _i_ [tr] _[,][ ∇]_ _[θ]_ _[L]_ [)]


In [50], the authors investigated the effectiveness of optimization-based metalearning in generalizing to similar but extrapolated tasks that are outside the
original task distribution _p_ ( _T_ ). The study found that, as task variability increases, black-box meta-learning methods such as SNAIL [40] and MetaNet

[41] acquire less generalizable learning strategies than gradient-based metalearning approaches like MAML.
However, despite its success, MAML faces some challenges that have motivated the development of other optimization-based meta-learning methods.
One of these challenges is the instability of MAML’s bi-level optimization. Fortunately, there are enhancements that can significantly improve optimization
process. For instance, Meta-SGD [51] and AlphaMAML [52] learn a vector of
learning rates _α_ automatically, rather than using a manually set scalar value _α_ .
Other methods like DEML [53], ANIL [54] and BOIL [55] suggest optimizing
only a subset of the parameters during adaptation. Additionally, MAML++

[56] proposes various modifications to stabilize the optimization process and
further improve the generalization performance. Moreover, Bias-transformation


11


Figure 4: Visual representation of the computation graph of MAML.


[50] and CAVIA [57] introduce context variables for increased expressive power,
while [58] enforces a well-conditioned parameter space based on the concepts
of the condition number [59].
Another significant challenge in MAML is the computationally expensive
process of backpropagating through multiple gradient adaptation steps. To
overcome this challenge, first-order alternatives to MAML such as FOMAML
and Reptile have been introduced [60]. For example, Reptile aims to find an initialization _θ_ that is close to each task’s optimal parameters. Another approach
is to optimize only the parameters of the last layer. For instance, [61] and [62]
perform a closed-form or convex optimization on top of meta-learned features.
Another solution is iMAML [63], which computes the full meta-gradient without differentiating through the optimization path, using the implicit function
theorem.


**4.3** **Meta-learning via distance metric learning**


In the context of low data regimes, such as in few-shot learning, simple nonparametric methods such as Nearest Neighbors [64] can be effective. However,
black-box and optimization-based meta-learning approaches discussed so far
in sections 4.1 and 4.2 have focused on using parametric base models, such as
neural networks. In this section we discuss meta-learning approaches that employ a non-parametric learning procedure. The key concept is to use parametric
meta-learners to produce effective non-parametric learners, thus eliminating
the need for second-order optimization, as required by several methods dis

12


Figure 5: Meta-learning via distance metric learning using Matching Network

[67].


**Algorithm 3** Meta-learning via metric learning (Matching Networks)

1: Randomly initialize _θ_
2: **while** not done **do**

3: Sample a task _T_ _i_ _∼_ _p_ ( _T_ ) _(or a mini-batch of tasks)_
4: Sample disjoint datasets _D_ _i_ [tr] _[,][ D]_ _i_ [ts] [from] _[ T]_ _[i]_
5: Compute ˆ _y_ [ts] = � _f_ _θ_ ( _x_ [ts] _, x_ _k_ ) _y_ _k_

( _x_ _k_ _,y_ _k_ ) _∈D_ _i_ [tr]

6: Update _θ_ using _∇_ _θ_ _L_ (ˆ _y_ [ts] _, y_ [ts] )

7: **end while**

8: **return** _θ_


cussed in section 4.2.
Suppose we are given a small training dataset _D_ _i_ [tr] [that presents a 1-shot-] _[N]_ [-]
way classification problem, i.e., _N_ classes with only one labeled data-point per
class, along with a test data-point _x_ [ts] . To classify _x_ [ts], a Nearest Neighbor learner
compares it with each training data-point in _D_ _i_ [tr] [. However, determining an ef-]
fective space and distance metric for this comparison can be challenging. For
example, using the _L_ 2 distance in pixel space for image data may not yield satisfactory results [65]. To overcome this, a distance metric can be derived by learning how to compare instances using meta-training data. To learn an appropriate
distance metric for comparing instances, a Siamese network [66] can be trained
to solve a binary classification problem that predicts whether two images belong to the same class. During meta-test time, each image in _D_ _i_ [tr] [is compared]
with the test image _x_ [ts] to determine whether they belong to the same class or
not. However, there is a nuance due to the mismatch between the binary classification problem during meta-training and the _N_ -way classification problem
during meta-testing. Matching Networks, introduced in [67], address this by
learning an embedding space with a network _f_ _θ_ and using Nearest Neighbors
in the learned space, as shown in Figure 5. The network is trained end-to-end to
ensure that meta-training is consistent with meta-testing. Algorithm 3 outlines
the meta-training process used by Matching Networks. It is similar to Algorithms 1 and 2, except that the base model is non-parametric, so there is no _ϕ_ _i_
(see lines 5 and 6).
However, Matching Networks are specifically designed for 1-shot classifica

13


Figure 6: Prototypical networks.


tion and cannot be directly applied to _K_ -shot classification problems (where
there are _K_ labeled samples per class). To address this issue, other methods,
such as Prototypical Networks [68], have been proposed. Prototypical Networks aggregate class information to create a prototypical embedding, as illustrated in Figure 6. In Prototypical Networks, line 5 of Algorithm 3 is replaced
with:

exp ( _−∥f_ _θ_ ( _x_ ) _−_ _c_ _l_ _∥_ )
_p_ _θ_ ( _y_ = _l|x_ ) =
~~�~~ _l_ _[′]_ [ exp (] _[−∥][f]_ _[θ]_ [(] _[x]_ [)] _[ −]_ _[c]_ _[l]_ _[′]_ _[∥]_ [)] _[,]_


where _c_ _l_ is the mean embedding of all the samples in the _l_ -th class, i.e., _c_ _l_ =
_K_ 1 � ( _x,y_ ) _∈D_ _i_ [tr] [1] [(] _[y]_ [ =] _[ l]_ [)] _[f]_ _[θ]_ [(] _[x]_ [)][.]

While methods such as Siamese networks, Matching Networks, and Prototypical Networks can perform few-shot classification by embedding data and
applying Nearest Neighbors [66, 67, 68], they may not be sufficient to capture complex relationships between data-points. To address this, alternative
approaches have been proposed. RelationNet [69] introduces a non-linear relation module that can reason about complex relationships between embeddings.
Garcia et al. [70] propose to use graph neural networks to perform message
passing on embeddings, allowing for the capture of more complex dependencies. Finally, Allen et al. [71] extend Prototypical Networks to learn an infinite
mixture of prototypes, which improves the model’s ability to represent the data
distribution.


**4.4** **Hybrid approaches**


Black-box, optimization-based, and distance metric-based meta-learning approaches define _F_ _θ_ ( _D_ _i_ [tr] _[, x]_ [ts] [)][ differently, but these approaches are not mutually]
exclusive and they can be combined in various ways. For instance, in [72], gradient descent is applied while conditioning on the data, allowing the model
to modulate the feature representations and capture inter-class dependencies.
In [73], LEO (Latent Embedding Optimization) combines optimization-based
meta-learning with a latent embedding produced by the RelationNet embedding proposed in [69]. The parameters of the model are first conditioned on
the input data and then further adapted through gradient descent. In [74], the
strength of both MAML and Prototypical Networks are combined to form a hybrid approach called Proto-MAML. This approach exploits the flexible adaptation of MAML, while initializing the last layer with ProtoNet to provide a simple
inductive bias that is effective for very-few-shot learning. Similarly, [75] proposes a model where the meta-learner operates using an optimization-based
meta-model, while the base learner exploits a metric-based approach (either


14


Matching Network or Prototypical Network). The distance metrics used by the
base learner can better adapt to different tasks thanks to the weight prediction
from the meta-learner.
In summary, researchers have explored combining black-box, optimizationbased, and distance metric-based meta-learning approaches to take advantage
of their individual strengths. These combined approaches aim to improve performance, adaptability, and generalization in few-shot learning tasks by integrating different methodologies.

### **5 Advanced meta-learning topics**


The field of meta-learning has seen rapid development in recent years, with
numerous methods proposed for learning to learn from a few examples. In
this section, we delve into advanced topics in meta-learning that extend the
meta-learning paradigm to more complex scenarios. We explore meta-learning
from multi-modal task distributions, the challenge of out-of-distribution tasks,
and unsupervised meta-learning. Additionally, we examine the relationship
between meta-learning and personalized federated learning, domain adaptation/generalization, as well as the intersection between meta-learning and continual learning. By delving into these advanced topics, we can gain a deeper
understanding of the potential of meta-learning and its applications in more
complex real-world scenarios.


**5.1** **Meta-learning from multimodal task distributions**


Meta-learning methods have traditionally focused on optimizing performance
within a unimodal task distribution _p_ ( _T_ ), assuming that all tasks are closely
related and share similarities within a single application domain. However,
recent studies have highlighted the limitations of standard meta-learning approaches when faced with significantly different tasks [76, 77, 78, 79]. In realworld scenarios, tasks are often diverse and sampled from a more complex task
distribution with multiple unknown modes. The performance of most metalearning approaches tends to deteriorate as the dissimilarity among tasks increases, indicating that a globally shared set of meta-parameters _θ_ may not adequately capture the heterogeneity among tasks and enable fast adaptation.
To address this challenge, MMAML [80] builds upon the standard MAML
approach by estimating the mode of tasks sampled from a multimodal task
distribution _p_ ( _T_ ) and adjusting the initial model parameters accordingly. Another approach proposed in [81] involves learning a meta-regularization conditioned on additional task-specific information. However, obtaining such additional task information may not always be feasible. Alternatively, some methods propose learning multiple model initializations _θ_ 1 _, θ_ 2 _, · · ·, θ_ _M_ and selecting
the most suitable one for each task, leveraging clustering techniques applied in
either the task-space or parameter-space [82, 83, 84, 85], or relying on the output
of an additional network. CAVIA [57] partitions the initial model parameters
into shared parameters across all tasks and task-specific context parameters,
while LGM-Net [86] directly generates classifier weights based on an encoded
task representation.


15


A series of related works (but outside of the meta-learning field) aim to
build a “universal representation” that encompasses a robust set of features capable of achieving strong performance across multiple datasets (or modes) [87,
88, 89, 42, 90, 90, 91]. This representation is subsequently adapted to individual
tasks in various ways. However, these approaches are currently limited to classification problems and do not leverage meta-learning techniques to efficiently
adapt to new tasks.
A more recent line of research focuses on cross-domain meta-learning, where
knowledge needs to be transferred from tasks sampled from a potentially multimodal distribution _p_ ( _T_ ) to target tasks sampled from a different distribution.
One notable study, BOIL [55], reveals that the success of meta-learning methods, such as MAML, can be attributed to large changes in the representation
during task learning. The authors emphasize the importance of updating only
the body (feature extractor) of the model and freezing the head (classifier) during the adaptation phase for effective cross-domain adaptation. Building on this
insight, DAML [92] introduces tasks from both seen and pseudo-unseen domains during meta-training to obtain domain-agnostic initial parameters capable of adapting to novel classes in unseen domains. In [93], the authors propose
a transferable meta-learning algorithm with a _meta task adaptation_ to minimize
the domain divergence and thus facilitate knowledge transfer across domains.
To further improve the transferability of cross-domain knowledge, [94] and

[95] propose to incorporate semi-supervised techniques into the meta-learning
framework. Specifically, [94] combines the representation power of large pretrained language models (e.g., BERT [31]) with the generalization capability
of prototypical networks enhanced by SMLMT [96] to achieve effective generalization and adaptation to tasks from new domains. In contrast, [95] promotes
the idea of task-level self-supervision by leveraging multiple views or augmentations of tasks.


**5.2** **Meta-learning & personalized federated learning**


Federated learning (FL) is a distributed learning paradigm where multiple clients
collaborate to train a shared model while preserving data privacy by keeping
their data locally stored. FedAvg [97] is a pioneering method that combines
local stochastic gradient descent on each client with model averaging on a central server. This approach performs well when local data across clients is independent and identically distributed (IID). However, in scenarios with heterogeneous (non-IID) data distributions, regularization techniques [98, 99, 100]
have been proposed to improve local learning.
Personalized federated learning (PFL) is an alternative approach that aims
to develop customized models for individual clients while leveraging the collaborative nature of FL. Popular PFL methods include L2GD [101], which combines local and global models, as well as multi-task learning methods like pFedMe

[102], Ditto [103], and FedPAC [104]. Clustered or group-based FL approaches

[105, 106, 107, 108] learn multiple group-based global models. In contrast,
meta-learning-based methods interpret PFL as a meta-learning algorithm, where
_personalization to a client_ aligns with _adaptation to a task_ [109]. Notably, various combinations of MAML-type methods with FL architectures have been explored in [110, 109, 111] to find an initial shared point that performs well after
personalization to each client’s local dataset. Additionally, the authors of [112]


16


Figure 7: Unsupervised pre-training.


proposed ARUBA, a meta-learning algorithm inspired by online convex optimization, which enhances the performance of FedAvg.
To summarize, there is a growing focus on addressing FL challenges in nonIID data settings. The integration of meta-learning has shown promising outcomes, leading to enhanced personalization and performance in PFL methods.


**5.3** **Unsupervised meta-learning with tasks construction**


In meta-training, constructing tasks typically relies on labeled data. However,
real-world scenarios often involve mostly, or only, unlabeled data, requiring
techniques that leverage unlabeled data to learn valuable feature representations that can transfer to downstream tasks with limited labeled data. One alternative to address this is through “self-supervised learning” (also known as
“unsupervised pre-training”) [113, 114, 115]. This involves training a model on
a large unlabeled dataset, as depicted in Figure 7, to capture informative features. Contrastive learning [116, 113] is commonly used in this context, aiming
to learn features by bringing similar examples closer together while pushing
differing examples apart. The learned features can then be fine-tuned on a target task _T_ new with limited labeled data _D_ new [tr] [, leading to improved performance]
compared to training from scratch. Another promising alternative is “unsupervised meta-learning,” which aims to automatically construct diverse and structured training tasks from unlabeled data. These tasks can then be used with any
meta-learning algorithm, such as MAML [8] and ProtoNet [68]. In this section,
we will explore methods for meta-training without predefined tasks and investigate strategies for automatically constructing tasks for meta-learning.
The method proposed in [117] constructs tasks based on unsupervised representation learning methods such as BiGAN [118, 119] or DeepCluster [120]
and clusters the data in the embedding space to assign pseudo-labels and construct tasks. Other methods such as UMTRA [121] and LASIUM [122] generate synthetic samples using image augmentations or pre-trained generative
networks. In particular, the authors in [121] construct a task _T_ _i_ for a 1-shot _N_ way classification problem by creating a support set _D_ _i_ [tr] [and a query set] _[ D]_ _i_ [ts] [as]
follows:


  - Randomly sample _N_ images and assign labels 1 _, . . ., N_, storing them in
_D_ _i_ [tr] [.]


17


 - Augment [1] each image in _D_ _i_ [tr] [, and store the resulting (augmented) images]
in _D_ _i_ [ts] [.]


Such augmentations can be based on domain knowledge or learned augmentation strategies like those proposed in [123]. In principle, task construction techniques can be applied beyond image-based augmentation. For instance, temporal aspects can be leveraged by incorporating time-contrastive learning on
videos, as demonstrated in [124]. Another approach is offered by Viewmaker
Networks [125], which learn augmentations that yield favorable outcomes not
only for images but also for speech and sensor data. Contrary to these works
focusing on generating pseudo tasks, Meta-GMVAE [126] and Meta-SVEBM

[127] address the problem by using variational autoencoders [128] and energybased models [129], respectively. However, these methods are limited to the
pseudo-labeling strategies used to create tasks, they rely on the quality of generated samples and they cannot scale to large-scale datasets.
To overcome this limitation, recent approaches have investigated the possibility of using self-supervised learning techniques to improve unsupervised
meta-learning methods. In particular, in [130], the relationship between contrastive learning and meta-learning is explored, demonstrating that established
meta-learning methods can achieve comparable performance to contrastive learning methods, and that representations transfer similarly well to downstream
tasks. Inspired by these findings, the authors in [131] integrate contrastive
learning in a two-stage training paradigm consisting of sequential pre-training
and meta-training stages. Another work [132] interprets a meta-learning problem as a set-level problem and maximizes the agreement between augmented
sets using SimCLR [133]. Finally, PsCo [133] builds upon MoCo [114] by progressively improving pseudo-labeling and constructing diverse tasks in an online manner. These findings indicate the potential for leveraging existing advances in meta-learning to improve contrastive learning (and vice-versa).
To meta-learn with unlabeled text data, some methods use language modeling, as shown in [134] for GPT-3. Here, the support set _D_ _i_ [tr] [consists of a se-]
quence of characters, and the query set _D_ _i_ [ts] [consists of the subsequent sequence]
of characters. However, this approach may not be suitable for text classification
tasks, such as sentiment analysis or identifying political bias. In [96], an alternative approach (SMLMT) for self-supervised meta-learning for few-shot natural
language classification tasks is proposed. SMLMT involves masking out words
and classifying the masked word to construct tasks. The process involves: (1)
sampling a subset of _N_ unique words and assigning each word a unique ID as
its class label, (2) sampling _K_ + _Q_ sentences that contain each of the _N_ words
and masking out the corresponding word in each sentence, and (3) constructing the support set _D_ _i_ [tr] [and the query set] _[ D]_ _i_ [ts] [using the masked sentences and]
their corresponding word IDs. SMLMT (for unsupervised meta-learning) is
compared to BERT [31], a method that uses standard self-supervised learning
and fine-tuning. SMLMT outperforms BERT on some tasks and achieves at least
equal performance on others. Furthermore, Hybrid-SMLMT (semi-supervised
meta-learning, which involves meta-learning on constructed tasks and supervised tasks), is compared to MT-BERT [135] (multi-task learning on supervised


1 Various augmentation techniques, like flipping, cropping, or reflecting an image, typically preserve its label. Likewise, nearby image patches or adjacent video frames share similar characteristics
and are therefore assigned the same label.


18


tasks) and LEOPARD [136] (an optimization-based meta-learner that uses only
supervised tasks). The results show that Hybrid-SMLMT significantly outperforms these other methods.


**5.4** **Meta-learning & domain adaptation/generalization**


Domain shift is a fundamental challenge, where the distribution of the input
data changes between the training and test domains. To address this problem,
there is a growing interest in utilizing meta-learning techniques for more effective domain adaptation and domain generalization. These approaches aim to
enable models to quickly adapt to new domains with limited data or to train robust models that achieve better generalization on domains they have not been
explicitly trained on.


**Effective domain adaptation via meta-learning**


Domain adaptation is a form of transductive transfer learning that leverages
source domain(s) _p_ _S_ ( _x, y_ ) to achieve high performance on test data from a target domain _p_ _T_ ( _x, y_ ) _._ It assumes _p_ _S_ ( _y|x_ ) = _p_ _T_ ( _y|x_ ) but _p_ _S_ ( _x_ ) _̸_ = _p_ _T_ ( _x_ ), treating
_domains_ as a particular kind of _tasks_, with a task _T_ _i_ ≜ _{p_ _i_ ( _x_ ) _, p_ _i_ ( _y|x_ ) _, L_ _i_ _}_ and
a domain _d_ _i_ ≜ _{p_ _i_ ( _x_ ) _, p_ ( _y|x_ ) _, L}_ . For example, healthcare data from different
hospitals with varying imaging techniques or patient demographics can correspond to different domains. Domain adaptation is most commonly achieved
via feature alignment as in [137, 138] or via translation between domains using
CycleGAN [139] as in [140, 141, 142]. Other approaches focus on aligning the
feature distribution of multiple source domains with the target domain [143]
or they address the multi-target domain adaptation scenario [144, 145, 146]
by designing models capable of adapting to multiple target domains. However, these methods face limitations when dealing with insufficient labeled data
in the source domain or when quick adaptation to new target domains is required. Additionally, they assume the input-output relationship (i.e., _p_ ( _y|x_ ))
is the same across domains. To solve these problems, some methods [147,
148, 149, 145] combine meta-learning with domain adaptation. In particular,
ARM [147] leverages contextual information extracted from batches of unlabeled data to learn a model capable of adapting to distribution shifts.


**Effective domain generalization via meta-learning**


Domain generalization enables models to perform well on new and unseen domains without requiring access to their data, as illustrated in Figure 8. This is
particularly useful in scenarios where access to data is restricted due to real-time
deployment requirements or privacy policies. For instance, an object detection
model for self-driving cars trained on three types of roads may need to be deployed to a new road without any data from that domain. In contrast to domain
adaptation, which requires access to (unlabeled) data from a specific target domain during training to specialize the model, domain generalization belongs to
the inductive setting. Most domain generalization methods aim to train neural
networks to learn domain-invariant representations that are consistent across
domains. For instance, domain adversarial training [150] trains the network to


19


Figure 8: Domain generalization problem.


make predictions based on features that cannot be distinguished between domains. Another approach is to directly align the representations between domains using similarity metrics, such as in [151]. Data augmentation techniques
are also used to enhance the diversity of the training data and improve generalization across domains [152, 153, 154]. Another way to improve generalization
to various domains is to use meta-learning and applying the episodic training
paradigm typical of MAML [50], as in [155, 156, 157, 158, 159, 160, 161]. For
instance, MLDG [158] optimizes a model by simulating the train-test domain
shift during the meta-training phase. MetaReg [159] proposes to meta-learn
a regularization function that improves domain generalization. DADG [161]
contains a discriminative adversarial learning component to learn a set of general features and a meta-learning-based cross-domain validation component to
further enhance the robustness of the classifier.


**5.5** **Meta-learning & continual learning**


This section explores the application of meta-learning to continual learning,
where learners continually accumulate experience over time to more rapidly acquire new knowledge or skills. Continual learning scenarios can be divided into
task-incremental learning, domain-incremental learning, and class-incremental
learning, depending on whether task identity is provided at test time or must
be inferred by the algorithm [162]. In this section, we focus on approaches that
specifically address task/class-incremental learning.
Traditionally, meta-learning has primarily focused on scenarios where a batch
of training tasks is available. However, real-world situations often involve tasks
presented sequentially, allowing for progressive leveraging of past experience.
This is illustrated in Figure 9, and examples include tasks that progressively
increase in difficulty or build upon previous knowledge, or robots learning diverse skills in changing environments.
Standard online learning involves observing tasks in a sequential manner,
without any task-specific adaptation or use of past experience to accelerate adaptation. To tackle this issue, researchers have proposed various approaches, including memory-based methods [163, 164, 165], regularization-based methods [166, 167, 168] and dynamic architectural methods [169, 170, 171]. However, each of these methods has its own limitations, such as scalability, memory
inefficiency, time complexity, or the need for task-specific parameters. Meta

20


learning has emerged as a promising approach for addressing continual learning. In [172], the authors introduced ANML, a framework that meta-learns an
activation-gating function that enables context-dependent selective activation
within a deep neural network. This selective activation allows the model to focus on relevant knowledge and avoid catastrophic forgetting. Other approaches
such as MER [173], OML [174], and LA-MAML [175] use gradient-based metalearning algorithms to optimize various objectives such as gradient alignment,
inner representations, or task-specific learning rates and learn update rules that
avoid negative transfer. These algorithms enable faster learning over time and
enhanced proficiency in each new task.


Figure 9: Continual learning.

### **6 Open challenges & opportunities**


Meta-learning has been a promising area of research that has shown impressive results in various machine learning domains. However, there are still open
challenges that need to be addressed in order to further advance the field. In
this section, we discuss some of these challenges and categorize them into three
main groups. Addressing these open challenges can lead to significant advances in meta-learning, which could potentially lead to more generalizable
and robust machine learning models.


**6.1** **Addressing fundamental problem assumptions**


The first category of challenges pertains to the fundamental assumptions made
in meta-learning problems.
One such challenge is related to generalization to out-of-distribution tasks
and long-tailed task distributions. Indeed, adaptation becomes difficult when
the few-shot tasks observed at meta-test time are from a different task distribution than the ones seen during meta-training. While there have been some
attempts to address this challenge, such as in [176, 93], it still remains unclear
how to address it. Ideas from the domain generalization and robustness literature could provide some hints and potentially be combined with meta-learning
to tackle these long-tailed task distributions and out-of-distribution tasks. For
example, possible directions are to define subtle regularization techniques to
prevent the meta-parameters from being very specific to the distribution of the
training tasks, or use subtle task augmentation techniques to generate synthetic
tasks that cover a wider range of task variations.
Another challenge in this category involves dealing with the multimodality of data. While the focus has been on meta-training over tasks from a single
modality, the reality is that we may have multiple modalities of data to work


21


with. Human beings have the advantage of being able to draw upon multiple
modalities, such as visual imagery, tactile feedback, language, and social cues,
to create a rich repository of knowledge and make more informed decisions.
For instance, we often use language cues to aid our visual decision-making processes. Rather than developing a prior that only works for a single modality,
exploring the concept of learning priors across multiple modalities of data is
a fascinating area to pursue. Different modalities have different dimensionalities or units, but they can provide complementary forms of information. While
some initial works in this direction have been reported, including [177, 178,
179], there is still a long way to go in terms of capturing all of this rich prior
information when learning new tasks.


**6.2** **Providing benchmarks and real-world problems**


The second category of challenges is related to providing/improving benchmarks to better reflect real-world problems and challenges.
Meta-learning has shown promise in a diverse set of applications, including
few-shot land cover classification [180], few-shot dermatological disease diagnosis [176], automatically providing feedback on student code [16], one-shot
imitation learning [181], drug discovery [182], motion prediction [183], and
language generation [14], to mention but a few. However, the lack of benchmark datasets that accurately reflect real-world problems with appropriate levels of difficulty and ease of use is a significant challenge for the field. Several efforts have been made towards creating useful benchmark datasets, including Meta-Dataset [74], Meta-Album Dataset [184], NEVIS’22 [185], MetaWorld Benchmark [186], Visual Task Adaptation Benchmark [187], Taskonomy
Dataset [188], VALUE Benchmark [189], and BIG Bench [190]. However, further work is needed to ensure that the datasets are comprehensive and representative of the diversity of real-world problems that meta-learning aims to
address.
Some ways with which existing benchmarks can be improved to better reflect real-world problems and challenges in meta-learning are: (1) to increase
the diversity and complexity of tasks that are included; (2) to consider more realistic task distributions that can change over time; and (3) to include real-world
data that is representative of the challenges faced in real-world applications of
meta-learning. For example, including medical data, financial data, time-series
data, or other challenging types of data (besides images and text) can help improve the realism and relevance of benchmarks.
Furthermore, developing benchmarks that reflect these more realistic scenarios can help improve the generalization and robustness of algorithms. This
ensures that algorithms are tested on a range of scenarios and that they are
robust and generalizable across a wide range of tasks. Better benchmarks are
essential for progress in machine learning and AI, as they challenge current algorithms to find common structures, reflect real-world problems, and have a
significant impact in the real world.


**6.3** **Improving core algorithms**


The last category of challenges in meta-learning is centered around improving
the core algorithms.


22


One major obstacle is the large-scale bi-level optimization problem encountered in popular meta-learning methods such as MAML. The computational
and memory costs of such approaches can be significant, and there is a need
to make them more practical, particularly for very large-scale problems, like
_learning effective optimizers_ [191].
In addition, a deeper theoretical understanding of various meta-learning
methods and their performance is critical to driving progress and pushing the
boundaries of the field. Such insights can inform and inspire further advancements in the field and lead to more effective and efficient algorithms. To achieve
these goals, several fundamental questions can be explored, including: (1) Can
we develop theoretical guarantees on the sample complexity and generalization performance of meta-learning algorithms? Understanding these aspects
can help us design more efficient and effective meta-learning algorithms that
require less data or less tasks. (2) Can we gain a better understanding of the
optimization landscape of meta-learning algorithms? For instance, can we identify the properties of the objective function that make it easier or harder to optimize? Can we design optimization algorithms that are better suited to the
bi-level optimization problem inherent in various meta-learning approaches?
(3) Can we design meta-learning algorithms that can better incorporate taskspecific or domain-specific expert knowledge, in a principled way, to learn more
effective meta-parameters?
Addressing such questions could enhance the design and performance of
meta-learning algorithms, and help us tackle increasingly complex and challenging learning problems.

### **7 Conclusion**


In conclusion, the field of artificial intelligence (AI) has witnessed significant
advancements in developing specialized systems for specific tasks. However,
the pursuit of generality and adaptability in AI across multiple tasks remains a
fundamental challenge.
Meta-learning emerges as a promising research area that seeks to bridge this
gap by enabling algorithms to learn how to learn. Meta-learning algorithms offer the ability to learn from limited data, transfer knowledge across tasks and
domains, and rapidly adapt to new environments. This review paper has explored various meta-learning approaches that have demonstrated promising
results in applications with scarce data. Nonetheless, numerous challenges and
unanswered questions persist, calling for further investigation.
A key area of focus lies in unifying various fields such as meta-learning, selfsupervised learning, domain generalization, and continual learning. Integrating and collaborating across these domains can generate synergistic advancements and foster a more comprehensive approach to developing AI systems. By
leveraging insights and techniques from these different areas, we can construct
more versatile and adaptive algorithms capable of learning from multiple tasks,
generalizing across domains, and continuously accumulating knowledge over
time.
This review paper serves as a starting point for encouraging research in this
direction. By examining the current state of meta-learning and illuminating the
challenges and opportunities, we aim to inspire researchers to explore inter

23


disciplinary connections and contribute to the progress of meta-learning while
integrating it with other AI research fields. Through collective efforts and collaboration, we can surmount existing challenges and unlock the full potential
of meta-learning to address a broad spectrum of complex problems faced by
intelligent systems.

### **References**


[1] Shubhra Kanti Karmaker et al. “Automl to date and beyond: Challenges
and opportunities”. In: _ACM Computing Surveys (CSUR)_ 54.8 (2021),
pp. 1–36.


[2] Joaquin Vanschoren. “Meta-Learning”. In: _Automated Machine Learning:_
_Methods, Systems, Challenges_ . Ed. by Frank Hutter, Lars Kotthoff, and
Joaquin Vanschoren. Springer International Publishing, 2019, pp. 35–
61.


[3] Timothy Hospedales et al. “Meta-learning in neural networks: A survey”. In: _IEEE transactions on pattern analysis and machine intelligence_ 44.9
(2021), pp. 5149–5169.


[4] Ricardo Vilalta and Youssef Drissi. “A perspective view and survey of
meta-learning”. In: _Artificial intelligence review_ 18 (2002), pp. 77–95.


[5] Mike Huisman, Jan N Van Rijn, and Aske Plaat. “A survey of deep metalearning”. In: _Artificial Intelligence Review_ 54.6 (2021), pp. 4483–4541.


[6] Oriol Vinyals. _Talk: Model vs Optimization Meta Learning_ . Neural Information Processing Systems (NIPS’17), Dec. 2017. url: `[https://evolution.](https://evolution.ml/pdf/vinyals.pdf)`
`[ml/pdf/vinyals.pdf](https://evolution.ml/pdf/vinyals.pdf)` .


[7] Jake Snell, Kevin Swersky, and Richard Zemel. “Prototypical networks
for few-shot learning”. In: _Advances in neural information processing sys-_
_tems_ 30 (2017).


[8] Chelsea Finn, Pieter Abbeel, and Sergey Levine. “Model-agnostic metalearning for fast adaptation of deep networks”. In: _International confer-_
_ence on machine learning_ . PMLR. 2017, pp. 1126–1135.


[9] Ruiying Geng et al. “Induction Networks for Few-Shot Text Classification”. In: _Proceedings of the 2019 Conference on Empirical Methods in Natu-_
_ral Language Processing and the 9th International Joint Conference on Natural_
_Language Processing (EMNLP-IJCNLP)_ . 2019, pp. 3904–3913.


[10] Bin Liang et al. “Few-shot aspect category sentiment analysis via metalearning”. In: _ACM Transactions on Information Systems_ 41.1 (2023), pp. 1–
31.


[11] Jiatao Gu et al. “Meta-learning for low-resource neural machine translation”. In: _2018 Conference on Empirical Methods in Natural Language Pro-_
_cessing, EMNLP 2018_ . Association for Computational Linguistics. 2020,
pp. 3622–3631.


[12] Andrea Madotto et al. “Personalizing dialogue agents via meta-learning”.
In: _Proceedings of the 57th Annual Meeting of the Association for Computa-_
_tional Linguistics_ . 2019, pp. 5454–5459.


24


[13] Kun Qian and Zhou Yu. “Domain Adaptive Dialog Generation via Meta
Learning”. In: _Proceedings of the 57th Annual Meeting of the Association for_
_Computational Linguistics_ . 2019, pp. 2639–2649.


[14] Fei Mi et al. “Meta-learning for low-resource natural language generation in task-oriented dialogue systems”. In: _Proceedings of the 28th Inter-_
_national Joint Conference on Artificial Intelligence_ . 2019, pp. 3151–3157.


[15] Chelsea Finn et al. “One-shot visual imitation learning via meta-learning”.
In: _Conference on robot learning_ . PMLR. 2017, pp. 357–368.


[16] Mike Wu et al. “ProtoTransformer: A meta-learning approach to providing student feedback”. In: _arXiv preprint arXiv:2107.14035_ (2021).


[17] Ozan Sener and Vladlen Koltun. “Multi-task learning as multi-objective
optimization”. In: _Advances in neural information processing systems_ 31
(2018).


[18] Zhao Chen et al. “Gradnorm: Gradient normalization for adaptive loss
balancing in deep multitask networks”. In: _International conference on ma-_
_chine learning_ . PMLR. 2018, pp. 794–803.


[19] Alex Kendall, Yarin Gal, and Roberto Cipolla. “Multi-task learning using uncertainty to weigh losses for scene geometry and semantics”. In:
_Proceedings of the IEEE conference on computer vision and pattern recogni-_
_tion_ . 2018, pp. 7482–7491.


[20] Ishan Misra et al. “Cross-stitch networks for multi-task learning”. In:
_Proceedings of the IEEE conference on computer vision and pattern recogni-_
_tion_ . 2016, pp. 3994–4003.


[21] Sebastian Ruder et al. “Latent multi-task architecture learning”. In: _Pro-_
_ceedings of the AAAI Conference on Artificial Intelligence_ . Vol. 33. 01. 2019,
pp. 4822–4829.


[22] Yuan Gao et al. “Nddr-cnn: Layerwise feature fusing in multi-task cnns
by neural discriminative dimensionality reduction”. In: _Proceedings of_
_the IEEE/CVF conference on computer vision and pattern recognition_ . 2019,
pp. 3205–3214.


[23] Vincent Dumoulin et al. “Feature-wise transformations”. In: _Distill_ (2018).
https://distill.pub/2018/feature-wise-transformations. doi: `[10.23915/](https://doi.org/10.23915/distill.00011)`
`[distill.00011](https://doi.org/10.23915/distill.00011)` .


[24] Shikun Liu, Edward Johns, and Andrew J Davison. “End-to-end multitask learning with attention”. In: _Proceedings of the IEEE/CVF conference_
_on computer vision and pattern recognition_ . 2019, pp. 1871–1880.


[25] Mingsheng Long et al. “Learning multiple tasks with multilinear relationship networks”. In: _Advances in neural information processing systems_
30 (2017).


[26] Andrew Jaegle et al. “Perceiver IO: A General Architecture for Structured Inputs & Outputs”. In: _International Conference on Learning Repre-_
_sentations_ .


[27] Chris Fifty et al. “Efficiently identifying task groupings for multi-task
learning”. In: _Advances in Neural Information Processing Systems_ 34 (2021),
pp. 27503–27516.


25


[28] Yu Zhang and Qiang Yang. “A survey on multi-task learning”. In: _IEEE_
_Transactions on Knowledge and Data Engineering_ 34.12 (2021), pp. 5586–
5609.


[29] Michael Crawshaw. “Multi-task learning with deep neural networks: A
survey”. In: _arXiv preprint arXiv:2009.09796_ (2020).


[30] Minyoung Huh, Pulkit Agrawal, and Alexei A Efros. “What makes ImageNet good for transfer learning?” In: _arXiv preprint arXiv:1608.08614_
(2016).


[31] Jacob Devlin et al. “Bert: Pre-training of deep bidirectional transformers for language understanding”. In: _Proceedings of the 2019 Conference_
_of the North American Chapter of the Association for Computational Linguis-_
_tics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN,_
_USA, June 2-7, 2019, Volume 1 (Long and Short Papers)_ . Association for
Computational Linguistics, 2019, pp. 4171–4186.


[32] Aakanksha Chowdhery et al. “Palm: Scaling language modeling with
pathways”. In: _arXiv preprint arXiv:2204.02311_ (2022).


[33] Hugo Touvron et al. “Llama: Open and efficient foundation language
models”. In: _arXiv preprint arXiv:2302.13971_ (2023).


[34] OpenAI. “GPT-4 Technical Report”. In: _ArXiv_ abs/2303.08774 (2023).


[35] Ananya Kumar et al. “Fine-Tuning can Distort Pretrained Features and
Underperform Out-of-Distribution”. In: _International Conference on Learn-_
_ing Representations_ .


[36] Yoonho Lee et al. “Surgical Fine-Tuning Improves Adaptation to Distribution Shifts”. In: _NeurIPS 2022 Workshop on Distribution Shifts: Connect-_
_ing Methods and Applications_ .


[37] Jason Phang, Thibault F´evry, and Samuel R Bowman. “Sentence encoders on stilts: Supplementary training on intermediate labeled-data
tasks”. In: _arXiv preprint arXiv:1811.01088_ (2018).


[38] Jeremy Howard and Sebastian Ruder. “Universal Language Model Finetuning for Text Classification”. In: _Proceedings of the 56th Annual Meeting_
_of the Association for Computational Linguistics (Volume 1: Long Papers)_ .
2018, pp. 328–339.


[39] Adam Santoro et al. “Meta-learning with memory-augmented neural
networks”. In: _International conference on machine learning_ . PMLR. 2016,
pp. 1842–1850.


[40] Nikhil Mishra et al. “A simple neural attentive meta-learner”. In: _Inter-_
_national Conference on Learning Representations_ . 2018.


[41] Tsendsuren Munkhdalai and Hong Yu. “Meta networks”. In: _Interna-_
_tional conference on machine learning_ . PMLR. 2017, pp. 2554–2563.


[42] Marta Garnelo et al. “Conditional neural processes”. In: _International_
_conference on machine learning_ . PMLR. 2018, pp. 1704–1713.


[43] Sachin Ravi and Hugo Larochelle. “Optimization as a model for fewshot learning”. In: _International conference on learning representations_ . 2017.


26


[44] Marcin Andrychowicz et al. “Learning to learn by gradient descent by
gradient descent”. In: _Advances in neural information processing systems_ 29
(2016).


[45] Ke Li and Jitendra Malik. “Learning to Optimize”. In: _International Con-_
_ference on Learning Representations_ . 2017.


[46] Olga Wichrowska et al. “Learned optimizers that scale and generalize”.
In: _International Conference on Machine Learning_ . PMLR. 2017, pp. 3751–
3760.


[47] Albert Shaw et al. “Meta architecture search”. In: _Advances in Neural In-_
_formation Processing Systems_ 32 (2019).


[48] Dongze Lian et al. “Towards fast adaptation of neural architectures with
meta learning”. In: _International Conference on Learning Representations_ .
2019.


[49] Luca Franceschi et al. “Bilevel programming for hyperparameter optimization and meta-learning”. In: _International conference on machine learn-_
_ing_ . PMLR. 2018, pp. 1568–1577.


[50] Chelsea Finn and Sergey Levine. “Meta-learning and universality: Deep
representations and gradient descent can approximate any learning algorithm”. In: (2018).


[51] Zhenguo Li et al. “Meta-sgd: Learning to learn quickly for few-shot
learning”. In: _arXiv preprint arXiv:1707.09835_ (2017).


[52] Harkirat Singh Behl, Atılım G¨unes¸ Baydin, and Philip HS Torr. “Alpha
maml: Adaptive model-agnostic meta-learning”. In: _6th ICML Workshop_
_on Automated Machine Learning, Thirty-Sixth International Conference on_
_Machine Learning (ICML)_ . 2019.


[53] Fengwei Zhou, Bin Wu, and Zhenguo Li. “Deep meta-learning: Learning to learn in the concept space”. In: _arXiv preprint arXiv:1802.03596_
(2018).


[54] Aniruddh Raghu et al. “Rapid learning or feature reuse? towards understanding the effectiveness of maml”. In: _International conference on_
_learning representations_ . 2023.


[55] Jaehoon Oh et al. “Boil: Towards representation change for few-shot
learning”. In: _The International Conference on Learning Representations (ICLR)_ .
2021.


[56] Antreas Antoniou, Harrison Edwards, and Amos Storkey. “How to train
your MAML”. In: _International Conference on Learning Representations_ .
2018.


[57] Luisa Zintgraf et al. “Fast context adaptation via meta-learning”. In: _In-_
_ternational Conference on Machine Learning_ . PMLR. 2019, pp. 7693–7702.


[58] Markus Hiller, Mehrtash Harandi, and Tom Drummond. “On Enforcing
Better Conditioned Meta-Learning for Rapid Few-Shot Adaptation”. In:
_Advances in Neural Information Processing Systems_ (2022).


[59] Ian Goodfellow, Yoshua Bengio, and Aaron Courville. _Deep learning_ .
MIT press, 2016.


27


[60] Alex Nichol, Joshua Achiam, and John Schulman. “On first-order metalearning algorithms”. In: _arXiv preprint arXiv:1803.02999_ (2018).


[61] Luca Bertinetto et al. “Meta-learning with differentiable closed-form
solvers”. In: _International Conference on Learning Representations (ICLR),_
_2019_ . 2019.


[62] Kwonjoon Lee et al. “Meta-learning with differentiable convex optimization”. In: _Proceedings of the IEEE/CVF conference on computer vision and_
_pattern recognition_ . 2019, pp. 10657–10665.


[63] Aravind Rajeswaran et al. “Meta-learning with implicit gradients”. In:
_Advances in neural information processing systems_ 32 (2019).


[64] T. Cover and P. Hart. “Nearest neighbor pattern classification”. In: _IEEE_
_Transactions on Information Theory_ 13.1 (1967), pp. 21–27. doi: `[10.1109/](https://doi.org/10.1109/TIT.1967.1053964)`
`[TIT.1967.1053964](https://doi.org/10.1109/TIT.1967.1053964)` .


[65] Richard Zhang et al. “The unreasonable effectiveness of deep features
as a perceptual metric”. In: _Proceedings of the IEEE conference on computer_
_vision and pattern recognition_ . 2018, pp. 586–595.


[66] Gregory Koch, Richard Zemel, Ruslan Salakhutdinov, et al. “Siamese
neural networks for one-shot image recognition”. In: _ICML deep learning_
_workshop_ . Vol. 2. 1. Lille. 2015.


[67] Oriol Vinyals et al. “Matching networks for one shot learning”. In: _Ad-_
_vances in neural information processing systems_ 29 (2016).


[68] Steinar Laenen and Luca Bertinetto. “On episodes, prototypical networks,
and few-shot learning”. In: _Advances in Neural Information Processing Sys-_
_tems_ 34 (2021), pp. 24581–24592.


[69] Flood Sung et al. “Learning to compare: Relation network for few-shot
learning”. In: _Proceedings of the IEEE conference on computer vision and_
_pattern recognition_ . 2018, pp. 1199–1208.


[70] Victor Garcia and Joan Bruna. “Few-shot learning with graph neural
networks”. In: _International Conference on Learning Representations_ . 2018.


[71] Kelsey Allen et al. “Infinite mixture prototypes for few-shot learning”.
In: _International Conference on Machine Learning_ . PMLR. 2019, pp. 232–
241.


[72] Xiang Jiang et al. “Learning to learn with conditional class dependencies”. In: _International conference on learning representations_ . 2019.


[73] Andrei A Rusu et al. “Meta-learning with latent embedding optimization”. In: (2019).


[74] Eleni Triantafillou et al. “Meta-dataset: A dataset of datasets for learning
to learn from few examples”. In: (2020).


[75] Duo Wang et al. “A hybrid approach with optimization-based and metricbased meta-learner for few-shot learning”. In: _Neurocomputing_ 349 (2019),
pp. 202–211.


[76] Yonglong Tian et al. “Rethinking few-shot image classification: a good
embedding is all you need?” In: _Computer Vision–ECCV 2020: 16th Euro-_
_pean Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XIV_
_16_ . Springer. 2020, pp. 266–282.


28


[77] Yinbo Chen et al. “Meta-Baseline: Exploring Simple Meta-Learning for
Few-Shot Learning”. In: _Proceedings of the IEEE/CVF International Con-_
_ference on Computer Vision (ICCV)_ . 2021, pp. 9062–9071.


[78] Yunhui Guo et al. “A broader study of cross-domain few-shot learning”.
In: _Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK,_
_August 23–28, 2020, Proceedings, Part XXVII 16_ . Springer. 2020, pp. 124–
141.


[79] Wei-Yu Chen et al. “A Closer Look at Few-shot Classification”. In: _Inter-_
_national Conference on Learning Representations_ . 2019.


[80] Risto Vuorio et al. “Multimodal model-agnostic meta-learning via taskaware modulation”. In: _Advances in neural information processing systems_
32 (2019).


[81] Giulia Denevi, Massimiliano Pontil, and Carlo Ciliberto. “The advantage of conditional meta-learning for biased regularization and fine tuning”. In: _Advances in Neural Information Processing Systems_ 33 (2020), pp. 964–
974.


[82] Huaxiu Yao et al. “Hierarchically structured meta-learning”. In: _Inter-_
_national Conference on Machine Learning_ . PMLR. 2019, pp. 7045–7054.


[83] Weisen Jiang, James Kwok, and Yu Zhang. “Subspace learning for effective meta-learning”. In: _International Conference on Machine Learning_ .
PMLR. 2022, pp. 10177–10194.


[84] Ghassen Jerfel et al. “Reconciling meta-learning and continual learning
with online mixtures of tasks”. In: _Advances in Neural Information Pro-_
_cessing Systems_ 32 (2019).


[85] Pan Zhou et al. “Task similarity aware meta learning: Theory-inspired
improvement on maml”. In: _Uncertainty in Artificial Intelligence_ . PMLR.
2021, pp. 23–33.


[86] Huaiyu Li et al. “LGM-Net: Learning to generate matching networks
for few-shot learning”. In: _International conference on machine learning_ .
PMLR. 2019, pp. 3825–3834.


[87] Lu Liu et al. “A Universal Representation Transformer Layer for FewShot Image Classification”. In: _International Conference on Learning Rep-_
_resentations_ . 2020.


[88] Wei-Hong Li, Xialei Liu, and Hakan Bilen. “Universal representation
learning from multiple domains for few-shot classification”. In: _Proceed-_
_ings of the IEEE/CVF International Conference on Computer Vision_ . 2021,
pp. 9526–9535.


[89] James Requeima et al. “Fast and flexible multi-task classification using
conditional neural adaptive processes”. In: _Advances in Neural Informa-_
_tion Processing Systems_ 32 (2019).


[90] Nikita Dvornik, Cordelia Schmid, and Julien Mairal. “Selecting relevant features from a multi-domain representation for few-shot classification”. In: _Computer Vision–ECCV 2020: 16th European Conference, Glas-_
_gow, UK, August 23–28, 2020, Proceedings, Part X 16_ . Springer. 2020, pp. 769–
786.


29


[91] Eleni Triantafillou et al. “Learning a universal template for few-shot
dataset generalization”. In: _International Conference on Machine Learning_ .
PMLR. 2021, pp. 10424–10433.


[92] Wei-Yu Lee, Jheng-Yu Wang, and Yu-Chiang Frank Wang. “DomainAgnostic Meta-Learning for Cross-Domain Few-Shot Classification”. In:
_ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and_
_Signal Processing (ICASSP)_ . IEEE. 2022, pp. 1715–1719.


[93] Bingyi Kang and Jiashi Feng. “Transferable Meta Learning Across Domains.” In: _UAI_ . 2018, pp. 177–187.


[94] Yue Li and Jiong Zhang. “Semi-supervised meta-learning for cross-domain
few-shot intent classification”. In: _Proceedings of the 1st Workshop on Meta_
_Learning and Its Applications to Natural Language Processing_ . 2021, pp. 67–
75.


[95] Wang Yuan et al. “Task-level Self-supervision for Cross-domain Fewshot Learning”. In: _Proceedings of the AAAI Conference on Artificial Intelli-_
_gence_ . Vol. 36. 3. 2022, pp. 3215–3223.


[96] Trapit Bansal et al. “Self-supervised meta-learning for few-shot natural
language classification tasks”. In: _Proceedings of the 2020 Conference on_
_Empirical Methods in Natural Language Processing (EMNLP)_ . 2020.


[97] Brendan McMahan et al. “Communication-efficient learning of deep
networks from decentralized data”. In: _Artificial intelligence and statistics_ .
PMLR. 2017, pp. 1273–1282.


[98] Tian Li et al. “Federated optimization in heterogeneous networks”. In:
_Proceedings of Machine learning and systems_ 2 (2020), pp. 429–450.


[99] Qinbin Li, Bingsheng He, and Dawn Song. “Model-contrastive federated learning”. In: _Proceedings of the IEEE/CVF Conference on Computer_
_Vision and Pattern Recognition_ . 2021, pp. 10713–10722.


[100] Sai Praneeth Karimireddy et al. “Scaffold: Stochastic controlled averaging for federated learning”. In: _International Conference on Machine Learn-_
_ing_ . PMLR. 2020, pp. 5132–5143.


[101] Filip Hanzely and Peter Richt´arik. “Federated learning of a mixture of
global and local models”. In: _arXiv preprint arXiv:2002.05516_ (2020).


[102] Canh T Dinh, Nguyen Tran, and Josh Nguyen. “Personalized federated
learning with moreau envelopes”. In: _Advances in Neural Information Pro-_
_cessing Systems_ 33 (2020), pp. 21394–21405.


[103] Tian Li et al. “Ditto: Fair and robust federated learning through personalization”. In: _International Conference on Machine Learning_ . PMLR. 2021,
pp. 6357–6368.


[104] Jian Xu, Xinyi Tong, and Huang Shao-Lun. “Personalized Federated Learning with Feature Alignment and Classifier Collaboration”. In: _Interna-_
_tional conference on learning representations_ . 2023.


[105] Avishek Ghosh et al. “An efficient framework for clustered federated
learning”. In: _Advances in Neural Information Processing Systems_ 33 (2020),
pp. 19586–19597.


30


[106] Moming Duan et al. “Flexible clustered federated learning for clientlevel data distribution shift”. In: _IEEE Transactions on Parallel and Dis-_
_tributed Systems_ 33.11 (2021), pp. 2661–2674.


[107] Felix Sattler, Klaus-Robert M¨uller, and Wojciech Samek. “Clustered federated learning: Model-agnostic distributed multitask optimization under privacy constraints”. In: _IEEE transactions on neural networks and learn-_
_ing systems_ 32.8 (2020), pp. 3710–3722.


[108] Lei Yang et al. “Personalized federated learning on non-IID data via
group-based meta-learning”. In: _ACM Transactions on Knowledge Discov-_
_ery from Data_ 17.4 (2023), pp. 1–20.


[109] Yihan Jiang et al. “Improving federated learning personalization via
model agnostic meta learning”. In: _arXiv preprint arXiv:1909.12488_ (2019).


[110] Alireza Fallah, Aryan Mokhtari, and Asuman Ozdaglar. “Personalized
federated learning with theoretical guarantees: A model-agnostic metalearning approach”. In: _Advances in Neural Information Processing Systems_
33 (2020), pp. 3557–3568.


[111] Fei Chen et al. “Federated meta-learning with fast convergence and efficient communication”. In: _arXiv preprint arXiv:1802.07876_ (2018).


[112] Mikhail Khodak, Maria-Florina F Balcan, and Ameet S Talwalkar. “Adaptive gradient-based meta-learning methods”. In: _Advances in Neural In-_
_formation Processing Systems_ 32 (2019).


[113] Ting Chen et al. “A simple framework for contrastive learning of visual
representations”. In: _International conference on machine learning_ . PMLR.
2020, pp. 1597–1607.


[114] Kaiming He et al. “Momentum contrast for unsupervised visual representation learning”. In: _Proceedings of the IEEE/CVF conference on com-_
_puter vision and pattern recognition_ . 2020, pp. 9729–9738.


[115] Jean-Bastien Grill et al. “Bootstrap your own latent-a new approach to
self-supervised learning”. In: _Advances in neural information processing_
_systems_ 33 (2020), pp. 21271–21284.


[116] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. “Representation learning with contrastive predictive coding”. In: _arXiv preprint arXiv:1807.03748_
(2018).


[117] Kyle Hsu, Sergey Levine, and Chelsea Finn. “Unsupervised learning via
meta-learning”. In: _International Conference on Learning Representations_ .
2019.


[118] Jeff Donahue, Philipp Kr¨ahenb¨uhl, and Trevor Darrell. “Adversarial feature learning”. In: _International Conference on Learning Representations_ .
2017.


[119] Jeff Donahue and Karen Simonyan. “Large scale adversarial representation learning”. In: _Advances in neural information processing systems_ 32
(2019).


[120] Mathilde Caron et al. “Deep clustering for unsupervised learning of visual features”. In: _Proceedings of the European conference on computer vision_
_(ECCV)_ . 2018, pp. 132–149.


31


[121] Siavash Khodadadeh, Ladislau Boloni, and Mubarak Shah. “Unsupervised meta-learning for few-shot image classification”. In: _Advances in_
_neural information processing systems_ 32 (2019).


[122] Siavash Khodadadeh et al. “Unsupervised meta-learning through latentspace interpolation in generative models”. In: _International Conference on_
_Learning Representations_ . 2021.


[123] Ekin D Cubuk et al. “Autoaugment: Learning augmentation strategies
from data”. In: _Proceedings of the IEEE/CVF conference on computer vision_
_and pattern recognition_ . 2019, pp. 113–123.


[124] Suraj Nair et al. “R3m: A universal visual representation for robot manipulation”. In: _Conference on Robot Learning_ . PMLR. 2023, pp. 892–909.


[125] Alex Tamkin, Mike Wu, and Noah Goodman. “Viewmaker networks:
Learning views for unsupervised representation learning”. In: _Interna-_
_tional Conference on Learning Representations_ . 2020.


[126] Dong Bok Lee et al. “Meta-gmvae: Mixture of gaussian vae for unsupervised meta-learning”. In: _International Conference on Learning Representa-_
_tions_ . 2021.


[127] Deqian Kong, Bo Pang, and Ying Nian Wu. “Unsupervised Meta-Learning
via Latent Space Energy-based Model of Symbol Vector Coupling”. In:
_Fifth Workshop on Meta-Learning at the Conference on Neural Information_
_Processing Systems_ . 2021.


[128] Diederik P Kingma and Max Welling. “Auto-encoding variational bayes”.
In: _International Conference on Learning Representations_ . 2014.


[129] Yee Whye Teh et al. “Energy-based models for sparse overcomplete representations”. In: _Journal of Machine Learning Research_ 4.Dec (2003), pp. 1235–
1260.


[130] Renkun Ni et al. “The close relationship between contrastive learning
and meta-learning”. In: _International Conference on Learning Representa-_
_tions_ . 2021.


[131] Zhanyuan Yang, Jinghua Wang, and Yingying Zhu. “Few-shot classification with contrastive learning”. In: _Computer Vision–ECCV 2022: 17th_
_European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part_
_XX_ . Springer. 2022, pp. 293–309.


[132] Dong Bok Lee et al. “Self-Supervised Set Representation Learning for
Unsupervised Meta-Learning”. In: _International Conference on Learning_
_Representations_ . 2023.


[133] Huiwon Jang, Hankook Lee, and Jinwoo Shin. “Unsupervised Metalearning via Few-shot Pseudo-supervised Contrastive Learning”. In: _In-_
_ternational Conference on Learning Representations_ . 2023.


[134] Tom Brown et al. “Language models are few-shot learners”. In: _Advances_
_in neural information processing systems_ 33 (2020), pp. 1877–1901.


[135] Chuhan Wu, Fangzhao Wu, and Yongfeng Huang. “One teacher is enough?
pre-trained language model distillation from multiple teachers”. In: _Find-_
_ings of the Association for Computational Linguistics: ACL-IJCNLP 2021_ .
2021, pp. 4408–4413.


32


[136] Trapit Bansal, Rishikesh Jha, and Andrew McCallum. “Learning to fewshot learn across diverse natural language classification tasks”. In: (2020),
pp. 5108–5123.


[137] Eric Tzeng et al. “Deep domain confusion: Maximizing for domain invariance”. In: _arXiv preprint arXiv:1412.3474_ (2014).


[138] Yaroslav Ganin et al. “Domain-adversarial training of neural networks”.
In: _The journal of machine learning research_ 17.1 (2016), pp. 2096–2030.


[139] Jun-Yan Zhu et al. “Unpaired image-to-image translation using cycleconsistent adversarial networks”. In: _Proceedings of the IEEE international_
_conference on computer vision_ . 2017, pp. 2223–2232.


[140] Kanishka Rao et al. “Rl-cyclegan: Reinforcement learning aware simulationto-real”. In: _Proceedings of the IEEE/CVF Conference on Computer Vision_
_and Pattern Recognition_ . 2020, pp. 11157–11166.


[141] Laura Smith et al. “Avid: Learning multi-stage tasks via pixel-level translation of human videos”. In: _arXiv preprint arXiv:1912.04443_ (2019).


[142] Judy Hoffman et al. “Cycada: Cycle-consistent adversarial domain adaptation”. In: _International conference on machine learning_ . Pmlr. 2018, pp. 1989–
1998.


[143] Han Zhao et al. “Adversarial multiple source domain adaptation”. In:
_Advances in neural information processing systems_ 31 (2018).


[144] Le Thanh Nguyen-Meidine et al. “Unsupervised multi-target domain
adaptation through knowledge distillation”. In: _Proceedings of the IEEE/CVF_
_Winter Conference on Applications of Computer Vision_ . 2021, pp. 1339–1347.


[145] Ziliang Chen et al. “Blending-target domain adaptation by adversarial
meta-adaptation networks”. In: _Proceedings of the IEEE/CVF Conference_
_on Computer Vision and Pattern Recognition_ . 2019, pp. 2248–2257.


[146] Behnam Gholami et al. “Unsupervised multi-target domain adaptation:
An information theoretic approach”. In: _IEEE Transactions on Image Pro-_
_cessing_ 29 (2020), pp. 3993–4002.


[147] Marvin Zhang et al. “Adaptive risk minimization: Learning to adapt to
domain shift”. In: _Advances in Neural Information Processing Systems_ 34
(2021), pp. 23664–23678.


[148] Wanqi Yang et al. “Few-Shot Unsupervised Domain Adaptation via Meta
Learning”. In: _2022 IEEE International Conference on Multimedia and Expo_
_(ICME)_ . IEEE. 2022, pp. 1–6.


[149] Yong Feng et al. “Similarity-based meta-learning network with adversarial domain adaptation for cross-domain fault identification”. In: _Knowledge-_
_Based Systems_ 217 (2021), p. 106829.


[150] Anthony Sicilia, Xingchen Zhao, and Seong Jae Hwang. “Domain adversarial neural networks for domain generalization: When it works and
how to improve”. In: _Machine Learning_ (2023), pp. 1–37.


[151] Baochen Sun and Kate Saenko. “Deep coral: Correlation alignment for
deep domain adaptation”. In: _Computer Vision–ECCV 2016 Workshops:_
_Amsterdam, The Netherlands, October 8-10 and 15-16, 2016, Proceedings,_
_Part III 14_ . Springer. 2016, pp. 443–450.


33


[152] Hongyi Zhang et al. “mixup: Beyond empirical risk minimization”. In:
_International Conference on Learning Representations_ . 2018.

[153] Vikas Verma et al. “Manifold mixup: Better representations by interpolating hidden states”. In: _International conference on machine learning_ .
PMLR. 2019, pp. 6438–6447.

[154] Huaxiu Yao et al. “Improving out-of-distribution robustness via selective augmentation”. In: _International Conference on Machine Learning_ . PMLR.
2022, pp. 25407–25437.

[155] Qi Dou et al. “Domain generalization via model-agnostic learning of
semantic features”. In: _Advances in Neural Information Processing Systems_
32 (2019).


[156] Da Li et al. “Episodic training for domain generalization”. In: _Proceed-_
_ings of the IEEE/CVF International Conference on Computer Vision_ . 2019,
pp. 1446–1455.

[157] Yiying Li et al. “Feature-critic networks for heterogeneous domain generalization”. In: _International Conference on Machine Learning_ . PMLR. 2019,
pp. 3915–3924.

[158] Da Li et al. “Learning to generalize: Meta-learning for domain generalization”. In: _Proceedings of the AAAI conference on artificial intelligence_ .
Vol. 32. 1. 2018.


[159] Yogesh Balaji, Swami Sankaranarayanan, and Rama Chellappa. “Metareg:
Towards domain generalization using meta-regularization”. In: _Advances_
_in neural information processing systems_ 31 (2018).

[160] Yang Shu et al. “Open domain generalization with domain-augmented
meta-learning”. In: _Proceedings of the IEEE/CVF Conference on Computer_
_Vision and Pattern Recognition_ . 2021, pp. 9624–9633.

[161] Keyu Chen, Di Zhuang, and J Morris Chang. “Discriminative adversarial domain generalization with meta-learning based cross-domain validation”. In: _Neurocomputing_ 467 (2022), pp. 418–426.

[162] Gido M van de Ven, Tinne Tuytelaars, and Andreas S Tolias. “Three
types of incremental learning”. In: _Nature Machine Intelligence_ 4.12 (2022),
pp. 1185–1197.

[163] David Lopez-Paz and Marc’Aurelio Ranzato. “Gradient episodic memory for continual learning”. In: _Advances in neural information processing_
_systems_ 30 (2017).

[164] Arslan Chaudhry et al. “Efficient lifelong learning with a-gem”. In: _In-_
_ternational Conference on Learning Representations_ . 2019.

[165] Sylvestre-Alvise Rebuffi et al. “icarl: Incremental classifier and representation learning”. In: _Proceedings of the IEEE conference on Computer Vision_
_and Pattern Recognition_ . 2017, pp. 2001–2010.

[166] Rahaf Aljundi, Marcus Rohrbach, and Tinne Tuytelaars. “Selfless sequential learning”. In: _International Conference on Learning Representa-_
_tions_ . 2019.


[167] James Kirkpatrick et al. “Overcoming catastrophic forgetting in neural
networks”. In: _Proceedings of the national academy of sciences_ 114.13 (2017),
pp. 3521–3526.


34


[168] Joan Serra et al. “Overcoming catastrophic forgetting with hard attention to the task”. In: _International Conference on Machine Learning_ . PMLR.
2018, pp. 4548–4557.


[169] Xilai Li et al. “Learn to grow: A continual structure learning framework
for overcoming catastrophic forgetting”. In: _International Conference on_
_Machine Learning_ . PMLR. 2019, pp. 3925–3934.


[170] Quang Pham et al. “Contextual transformation networks for online continual learning”. In: _International Conference on Learning Representations_ .
2021.


[171] Andrei A Rusu et al. “Progressive neural networks”. In: _arXiv preprint_
_arXiv:1606.04671_ (2016).


[172] Shawn Beaulieu et al. “Learning to continually learn”. In: _24th European_
_Conference on Artificial Intelligence_ . 2020.


[173] Matthew Riemer et al. “Learning to learn without forgetting by maximizing transfer and minimizing interference”. In: _International Confer-_
_ence on Learning Representations_ . 2019.


[174] Khurram Javed and Martha White. “Meta-learning representations for
continual learning”. In: _Advances in neural information processing systems_
32 (2019).


[175] Gunshi Gupta, Karmesh Yadav, and Liam Paull. “Look-ahead meta learning for continual learning”. In: _Advances in Neural Information Processing_
_Systems_ 33 (2020), pp. 11588–11598.


[176] Viraj Prabhu et al. “Few-shot learning for dermatological disease diagnosis”. In: _Machine Learning for Healthcare Conference_ . PMLR. 2019, pp. 532–
552.


[177] Paul Pu Liang et al. “Cross-modal generalization: Learning in low resource modalities via meta-alignment”. In: _Proceedings of the 29th ACM_
_International Conference on Multimedia_ . 2021, pp. 2680–2689.


[178] Jean-Baptiste Alayrac et al. “Flamingo: a visual language model for fewshot learning”. In: _Advances in Neural Information Processing Systems_ 35
(2022), pp. 23716–23736.


[179] Scott Reed et al. “A generalist agent”. In: _Transactions on Machine Learn-_
_ing Research_ (2022).


[180] Marc Rußwurm et al. “Meta-learning for few-shot land cover classification”. In: _Proceedings of the ieee/cvf conference on computer vision and pattern_
_recognition workshops_ . 2020, pp. 200–201.


[181] Tianhe Yu et al. “One-shot imitation from observing humans via domainadaptive meta-learning”. In: _Robotics: Science and Systems XIV_ (2018).


[182] Cuong Q Nguyen, Constantine Kreatsoulas, and Kim M Branson. “Metalearning gnn initializations for low-resource molecular property prediction”. In: _4th Lifelong Machine Learning Workshop at ICML 2020_ . 2020.


[183] Liang-Yan Gui et al. “Few-shot human motion prediction via meta-learning”.
In: _Proceedings of the European Conference on Computer Vision (ECCV)_ .
2018, pp. 432–450.


35


[184] Ihsan Ullah et al. “Meta-album: Multi-domain meta-dataset for few-shot
image classification”. In: _Advances in Neural Information Processing Sys-_
_tems_ 35 (2022), pp. 3232–3247.


[185] Jorg Bornschein et al. “NEVIS’22: A Stream of 100 Tasks Sampled from
30 Years of Computer Vision Research”. In: _arXiv preprint arXiv:2211.11747_
(2022).


[186] Tianhe Yu et al. “Meta-world: A benchmark and evaluation for multitask and meta reinforcement learning”. In: _Conference on robot learning_ .
PMLR. 2020, pp. 1094–1100.


[187] Xiaohua Zhai et al. “The visual task adaptation benchmark”. In: (2019).


[188] Amir R Zamir et al. “Taskonomy: Disentangling task transfer learning”.
In: _Proceedings of the IEEE conference on computer vision and pattern recog-_
_nition_ . 2018, pp. 3712–3722.


[189] Linjie Li et al. “Value: A multi-task benchmark for video-and-language
understanding evaluation”. In: _Thirty-fifth Conference on Neural Informa-_
_tion Processing Systems Datasets and Benchmarks Track (Round 1)_ . 2021.


[190] Aarohi Srivastava et al. “Beyond the imitation game: Quantifying and
extrapolating the capabilities of language models”. In: _Transactions on_
_Machine Learning Research_ (2023). issn: 2835-8856.


[191] Luke Metz et al. “Tasks, stability, architecture, and compute: Training
more effective learned optimizers, and using them to train themselves”.
In: _arXiv preprint arXiv:2009.11243_ (2020).


36


