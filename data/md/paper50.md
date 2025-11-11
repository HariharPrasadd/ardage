## **CTAB-GAN: Effective Table Data Synthesizing**



Zilong Zhao [∗]

Aditya Kunar [∗]

Z.Zhao-8@tudelft.nl
A.Kunar@student.tudelft.nl
TU Delft

Delft, Netherlands


Robert Birke

ABB Research Switzerland

Dättwil, Switzerland
robert.birke@ch.abb.com


**ABSTRACT**


While data sharing is crucial for knowledge development, privacy
concerns and strict regulation (e.g., European General Data Protection Regulation (GDPR)) unfortunately limit its full effectiveness.
Synthetic tabular data emerges as an alternative to enable data
sharing while fulfilling regulatory and privacy constraints. The
state-of-the-art tabular data synthesizers draw methodologies from
Generative Adversarial Networks (GAN) and address two main data
types in industry, i.e., continuous and categorical. In this paper, we
develop CTAB-GAN, a novel conditional table GAN architecture
that can effectively model diverse data types, including a mix of
continuous and categorical variables. Moreover, we address data
imbalance and long tail issues, i.e., certain variables have drastic
frequency differences across large values. To achieve those aims,
we first introduce the information loss and classification loss to the
conditional GAN. Secondly, we design a novel conditional vector,
which efficiently encodes the mixed data type and skewed distribution of data variable. We extensively evaluate CTAB-GAN with
the state of the art GANs that generate synthetic tables, in terms
of data similarity and analysis utility. The results on five datasets
show that the synthetic data of CTAB-GAN remarkably resembles
the real data for all three types of variables and results into higher
accuracy for five machine learning algorithms, by up to 17%.


**KEYWORDS**


GAN, data synthesis, tabular data, imbalanced distribution


**1** **INTRODUCTION**


“Data is the new oil” is a quote that goes back to 2006, which is
credited to mathematician Clive Humby. It has recently picked up
more steam after The Economist published a 2017 report [ 21 ] titled “The world’s most valuable resource is no longer oil, but data”.
Many companies nowadays discover valuable business insights
from various internal and external data sources. However, the big
knowledge behind big data often impedes personal privacy and
leads to unjustified analysis [ 17 ]. To prevent the abuse of data and
the risks of privacy breaching, the European Commission introduced the European General Data Protection Regulation (GDPR)
and enforced strict data protection measures. This however instills


∗ Both authors contributed equally to this research.



Hiek Van der Scheer

Aegon
Den Haag, Netherlands
hiek.vanderscheer@aegon.com


Lydia Y. Chen
Tu Delft

Delft, Netherlands
Y.Chen-10@tudelft.nl


a new challenge in the data-driven industries to look for new scientific solutions that can empower big discovery while respecting the
constraints of data privacy and governmental regulation.
An emerging solution is to leverage synthetic data [ 16 ], which
statistically resembles real data and can comply with GDPR due to
its synthetic nature. The industrial datasets (at stakeholders like
banks, insurance companies, and health care) present multi-fold
challenges. First of all, such datasets are organized in tables and
populated with both continuous and categorical variables, or a mix
of the two, e.g., the value of mortgage for a loan holder. This value
can be either 0 (no mortgage) or some continuous positive number.
Here, we term such a type of variables, **mixed variable** . Secondly
data variables often have a wide range of values as well as skewed
frequency distribution, e.g., the statistic of transaction amount for
credit card. Most transactions should be within 0 and 500 bucks

(i.e. daily shopping for food and clothes), but exceptions of a high
transaction amount surely exist.
Generative Adversarial Network (GAN) [ 7 ] is one of the emerging data synthesizing methodologies. The GAN is first trained on
a real dataset. Then used to generate data. Beyond its success on
generating images, GAN has recently been applied to generate
tabular data [ 16, 19, 24, 25 ]. The state of the art of tabular generators [ 24 ] treats categorical variables via conditional GAN, where
each categorical value is considered as a condition. However, their
focus is only on two types of variables, namely continuous and
categorical, overlooking an important class of mixed data type. In
addition, it is unclear if existing solutions can efficiently handle
highly imbalanced categorical variables and skewed continuous
variables.

In this paper, we aim to design a tabular data synthesizer that
addresses the limitations of the prior state-of-the-art: (i) encoding
mixed data type of continuous and categorical variables, (ii) efficient modeling of long tail continuous variables and (iii) increased
robustness to imbalanced categorical variables along with skewed
continuous variables. Hence, we propose a novel conditional table
generative adversarial network, CTAB-GAN. Two key features of
CTAB-GAN are the introduction of classification loss in conditional
GAN, and novel encoding for the conditional vector that efficiently
encodes mixed variables and helps to deal with highly skewed
distributions for continuous variables.


KDD, 2021, Singapore Zilong Zhao, Aditya Kunar, Hiek Van der Scheer, Robert Birke, and Lydia Y. Chen


**(a) Mortgage in Loan dataset [9]** **(b) Amount in Credit dataset [5]** **(c) Hours-per-week in Adult dataset [5]**


**Figure 1: Challenges of modeling industrial dataset using existing GAN-based table generator: (a) mixed type, (b) long tail**
**distribution, and (c) skewed data**



We rigorously evaluate CTAB-GAN in three dimensions: (i) utility of machine learning based analysis on the synthetic data, (ii)
statistical similarity to the real data, and (iii) privacy preservability. Specifically, the proposed CTAB-GAN is tested on 5 widely
used machine learning datasets: Adult, Covertype, Credit, Intrusion
and Loan, against 4 state-of-the-art GAN-based tabular data generation algorithms: CTGAN, TableGAN, CWGAN and MedGAN.
Our results show that CTAB-GAN not only outperforms all the
comparisons in machine learning utility and statistical similarity
but also provides better distance-based privacy guarantees than
TableGAN, the second best performing algorithm in the machine
learning utility and statistical similarly evaluation.
The main contributions of this study can be summarized as
follows:


  - Novel conditional adversarial network which introduces a

classifier providing additional supervision to improve its
utility for ML applications.

  - Efficient modelling of continuous, categorical, and **mixed**
variables via novel data encoding and conditional vector.

  - Light-weight data pre-processing to mitigate the impact of
long tail distribution of continuous variables.

  - Providing an effective data synthesizer for the relevant stakeholders.


**1.1** **Motivation**


In this subsection, we empirically demonstrate how the prior stateof-the-art methods fall short in solving challenges in industrial data
sets. The detailed experimental setup can be found in Sec. 4.1.
**Mixed data type variables** . To the best of our knowledge, existing GAN-based tabular generators only consider table columns as
either categorical or continuous. However, in reality, a variable can
be a mix of these two types, and often variables have missing values.



The _Mortgage_ variable from the Loan dataset is a good example of
mixed variable. Fig. 1a shows the distribution of the original and
synthetic data generated by 4 state-of-the-art algorithms for this
variable. According to the data description, a loan holder can either
have no mortgage (0 value) or a mortgage (any positive value). In
appearance this variable is not a categorical type due to the numeric
nature of the data. So all 4 state-of-the-art algorithms treat this variables as continuous type without capturing the special meaning
of the value zero. Hence, all 4 algorithms generate a value around
0 instead of exact 0. And the negative values for Mortgage have
no/wrong meaning in the real world.
**Long tail distributions** . Many real world data can have long
tail distributions where most of the occurrences happen near the
initial value of the distribution, and rare cases towards the end.
Fig. 1c plots the cumulative frequency for the original (top) and
synthetic (bottom) data generated by 4 state-of-the-art algorithms
for the _Amount_ in the Credit dataset. This variable represents the
transaction amount when using credit cards. One can imagine that
most transactions have small amounts, ranging from few bucks to
thousands of dollars. However, there definitely exists a very small
number of transactions with large amounts. Note that for ease of
comparison both plots use the same x-axis, but Real has no negative
values. Real data clearly has 99% of occurrences happening at the
start of the range, but the distribution extends until around 25000.
In comparison none of the synthetic data generators is able to learn
and imitate this behavior.

**Skewed multi-mode continuous variables** . The term _multi-_

_mode_ is extended from Variational Gaussian Mixtures (VGM). More
details are given in Sec. 3.3. The intuition behind using multiple
modes can be easily captured from Fig. 1c. The figure plots in each
row the distribution of the working _Hours-per-week_ variable from
the Adult dataset. This is not a typical Gaussian distribution. There


CTAB-GAN: Effective Table Data Synthesizing KDD, 2021, Singapore



is an obvious peak at 40 hours but with several other lower peaks,
e.g. at 50, 20 and 45. Also the number of people working 20 hours
per week is higher than those working 10 or 30 hours per week. This
behavior is difficult to capture for the state-of-the-art data generators (see subsequent rows in Fig.1c). The closest results are obtained
by CTGAN which uses Gaussian mixture estimation for continuous
variables. However, CTGAN loses some modes compared to the
original distribution.
The above examples show the shortcomings of current state-ofthe-art GAN-based tabular data generation algorithms and motivate
the design of our proposed CTAB-GAN.


**2** **RELATED STUDIES**


We divide the related studies using GAN to generate tabular data
into two categories: (i) based on GAN, and (ii) based on conditional
GAN.

**GAN-based generator** Several studies extend GAN to accommodate categorical variables by augmenting GAN architecture.
MedGAN [ 4 ] combines an auto-encoder with a GAN. It can generate continuous or discrete variables, and has been applied to
generate synthetic electronic health record (EHR) data. CrGANCnet [ 16 ] uses GAN to conduct Airline Passenger Name Record
Generation. It integrates the Cramér Distance [ 2 ] and Cross-Net
architecture [ 23 ] into the algorithm. In addition to generating with
continuous and categorical data types, CrGAN-Cnet can also handle
missing value in the table by adding new variables. TableGAN [ 19 ]
introduces information loss and a classifier into GAN framework.
It specifically adopts Convolutional Neural Network (CNN) for
generator, discriminator and classifier. Although aforementioned
algorithms can generate tabular data, they cannot specify how to
generate from a specific class for particular variable. For example,
it is not possible to generate health record for users whose sex is
female. In addition to data generation, privacy is another important
factor for synthetic tabular data. PATE-GAN [ 27 ] is not specifically
designed for tabular data generation, but it proposes a framework
which generates synthetic data with differential privacy guarantees.
**Conditional GAN-based generator** Due to the limitation of
controlling generated data via GAN, Conditional GAN is increasingly used, and its conditional vector can be used to specify to
generate a particular class of data. CW-GAN [ 6 ] applies the Wasserstein distance [ 1 ] into the conditional GAN framework. It leverages
the usage of conditional vector to oversample the minority class
to address imbalanced tabular data generation. CTGAN [ 24 ] integrates PacGAN [ 14 ] structure in its discriminator and uses WGAN
loss plus gradient penalty [ 8 ] to train a conditional GAN framework. It also adopts a strategy called training-by-sampling, which
takes advantage of conditional vector, to deal with the imbalanced
categorical variable problem.
In our paper, we not only focus on modelling continuous or categorical variables, but also cover the mixed data type (i.e., variables
that contain both categorical and continuous values, or even missing values). We effectively combine the strengths of prior art, such
as classifier, information loss, effective encoding, and conditional
vector. Furthermore, we proactively address the pain point of long
tail variable distributions and propose a new conditional vector
structure to better deal with imbalanced datasets.



**3** **CTAB-GAN**


CTAB-GAN is a tabular data generator designed to overcome the
challenges outlined in Sec. 1.1. In CTAB-GAN we invent a _Mixed-_
_type Encoder_ which can better represent mixed categorical-continuous
variables as well as missing values. CTAB-GAN is based on a conditional GAN (CGAN) to efficiently treat minority classes [ 6, 24 ],
with the addition of classification and information loss [ 18, 19 ]
to improve semantic integrity and training stability, respectively.
Finally, we leverage a log-frequency sampler to overcome the mode
collapse problem for imbalanced variables.


**3.1** **Technical background**


GANs are a popular method to generate synthetic data first applied
with great success to images [ 10, 11 ] and later adapted to tabular
data [ 26 ]. GANs leverage an adversarial game between a generator
trying to synthesize realistic data and a discriminator trying to
discern synthetic from real samples.
To address the problem of dataset imbalance, we leverage _condi-_
_tional generator_ and _training-by-sampling_ methods [ 24 ]. The idea
behind this is to use an additional vector, termed as the conditional
vector, to represent the classes of categorical variables. This vector
is both fed to the generator and used to bound the sampling of
the real training data to subsets satisfying the condition. We can
leverage the condition to resample all classes giving higher chances
to minority classes to train the model.
To enhance the generation quality, we incorporate two extra
terms in the loss function of the generator [ 18, 19 ]: information and
classification loss. The information loss penalizes the discrepancy
between statistics of the generated data and the real data. This helps
to generate data which is statistically closer to the real one. The
classification loss requires to add to the GAN architecture an auxiliary classifier in parallel to the discriminator. For each synthesized
label the classifier outputs a predicted label. The classification loss
quantifies the discrepancy between the synthesized and predicted
class. This helps to increase the semantic integrity of synthetic
records. For instance, (sex=female, disease=prostate cancer) is not
a semantically correct record as women do not have a prostate, and
no such record should appear in the original data and is hence not
learnt by the classifier.
To counter complex distributions in continuous variables we
embrace the _mode-specific normalization_ idea [ 24 ] which encodes
each value as a value-mode pair stemming from Gaussian mixture
model.


**3.2** **Design of CTAB-GAN**


The structure of CTAB-GAN comprises three blocks: Generator G,
Discriminator D and an auxiliary Classifier C (see Fig. 2). Since
our algorithm is based on conditional GAN, the generator requires
a noise vector plus a conditional vector. Details on the conditional
vector are given in Sec. 3.4. To simplify the figure, we omit the
encoding and decoding of the synthetic and real data detailed in
Sec. 3.3.

GANs are trained via a zero-sum minimax game where the discriminator tries to maximize the objective, while the generator tries
to minimize it. The game can be seen as a mentor ( D ) providing
feedback to a student ( G ) on the quality of his work. Here, we


KDD, 2021, Singapore Zilong Zhao, Aditya Kunar, Hiek Van der Scheer, Robert Birke, and Lydia Y. Chen


**Figure 2: Synthetic Tabular Data Generation via CTAB-GAN**


distribution of a mixed variable shown in red in Fig. 3a. One can
see that values can either be exactly _𝜇_ 0 or _𝜇_ 3 (the categorical part)
or distributed around two peaks in _𝜇_ 1 and _𝜇_ 2 (the continuous part).
We treat the continuous part using a variational Gaussian mixture
model (VGM) [ 3 ] to estimate the number of modes _𝑘_, e.g. _𝑘_ = 2
in our example, and fit a Gaussian mixture. The learned Gaussian

mixture is:



**(a) Mixed type variable distribution**
**with VGM**



**(b) Mode selection of single value in**
**continuous variable**



P =



2
∑︁ _𝜔_ _𝑘_ N ( _𝜇_ _𝑘_ _, 𝜎_ _𝑘_ ) (1)

_𝑘_ =1



**Figure 3: Encoding for mix data type variable**


introduce additional feedback for G based on the information loss
and classification loss. The information loss matches the first-order
(i.e., mean) and second-order (i.e., standard deviation) statistics
of synthesized and real records. This leads the synthetic records
to have the same statistical characteristics as the real records. In

addition, the classifier is trained to learn the correlation between
classes and the other variable values using the real training data.
The classification loss helps to check the semantic integrity, and
penalizes synthesized records where the combination of values are
semantically incorrect. These two losses are added to the original
loss term of G during training.
G and D are implemented by a four and a two layers CNN,
respectively. CNNs are good at capturing the relation between
pixels within an image [ 12 ], which in our case, can help to increase
the semantic integrity of synthetic data. C uses a 7 layers MLP.
The classifier is trained on the original data to better interpret the
semantic integrity. Hence synthetic data are reverse transformed
from their encoding (details in Sec. 3.3) before being used as input
for C to create the class label predictions.


**3.3** **Mixed-type Encoder**


The tabular data is encoded variable by variable. We distinguish
three types of variables: categorical, continuous and mixed. We
define variables as mixed if they contain both categorical and continuous values or continuous values with missing values. We propose the new Mixed-type Encoder to deal with such variables. With
this encoder, values of mixed variables are seen as concatenated
value-mode pairs. We illustrate the encoding via the exemplary



where N is the normal distribution and _𝜔_ _𝑘_, _𝜇_ _𝑘_ and _𝜎_ _𝑘_ are the
weight, mean and standard deviation of each mode, respectively.
To encode values in the continuous region of the variable distribution, we associate and normalize each value with the mode
having the highest probability (see Fig. 3b). Given _𝜌_ 1 and _𝜌_ 2 being
the probability density from the two modes in correspondence of
the variable value _𝜏_ to encode, we select the mode with the highest probability. In our example _𝜌_ 1 is higher and we use mode 1 to
normalize _𝜏_ . The normalized value _𝛼_ is:


_𝛼_ = _[𝜏]_ [−] _[𝜇]_ [1] (2)

4 _𝜎_ 1


Moreover we keep track of the mode _𝛽_ used to encode _𝜏_ via one
hot encoding, e.g. _𝛽_ = [ 0 _,_ 1 _,_ 0 _,_ 0 ] in our example. The final encoding
is giving by the concatenation of _𝛼_ and _𝛽_ : _𝛼_ [�] _𝛽_ where [�] is the
vector concatenation operator.
The categorical values are treated similarly, except _𝛼_ directly
represents the value of the mode, e.g. corresponding to _𝜇_ 0 or _𝜇_ 3 in
our example. Hence, for a value in _𝜇_ 3, the final encoding is given
by _𝜇_ 3 �[ 0 _,_ 0 _,_ 0 _,_ 1 ] . Note that categorical values are not limited to
numbers. They can be of any type such as a string or even missing.
We can map these symbols to a numeric value outside of the range
of the continuous region.
Categorical variables use the same encoding as the continuous
intervals of mixed variables. Categorical variables are encoded via
a one-hot vector _𝛾_ . Missing values are treated as a separate unique
class and we add an extra bit to the one-hot vector for it. A row with

[ 1 _, . . ., 𝑁_ ] variables is encoded by concatenation of the encoding
of all variable values, i.e. either ( _𝛼_ [�] _𝛽_ ) for continuous and mixed
variables or _𝛾_ for categorical variables. Having _𝑛_ continuous/mixed
variables and _𝑚_ categorical variables ( _𝑛_ + _𝑚_ = _𝑁_ ) the final encoding


CTAB-GAN: Effective Table Data Synthesizing KDD, 2021, Singapore


**Table 1: Description of datasets. The notations** _𝐶_ **,** _𝐵_ **,** _𝑀_ **&**
_𝑀𝑖_ **represent number of continuous, binary, multi-class cat-**
**egorical, mixed variables and imbalance ratios (i.e no. of mi-**
**nority samples/no. of majority samples) respectively.**



**Figure 4: Conditional vector: example selects class 2 from**
**third variable out of three**







is:

_𝑛_
� _𝛼_ _𝑖_ � _𝛽_ _𝑖_

_𝑖_ =1



_𝑁_
� _𝛾_ _𝑗_ (3)

_𝑗_ = _𝑛_ +1



**3.4** **Counter imbalanced training datasets**


In CTAB-GAN, we use conditional GAN to counter imbalanced
training datasets. When we sample real data, we use the conditional
vector to filter and rebalance the training data.
The conditional vector V is a bit vector given by the concatenation of all mode one-hot encodings _𝛽_ (for continuous and mixed
variables) and all class one-hot encodings _𝛾_ (for categorical variables) for all variables present in Eq. (3) . Each conditional vector
specifies a single mode or a class. More in detail, V is a zero vector
with a single one in correspondence to the selected variable with
selected mode/class. Fig. 4 shows an example with three variables,
one continuous ( _𝐶_ 1 ), one mixed ( _𝐶_ 2 ) and one categorical ( _𝐶_ 3 ), with
class 2 selected on _𝐶_ 3 .
To rebalance the dataset, each time we need a conditional vector
during training, we first randomly choose a variable with uniform
probability. Then we calculate the probability distribution of each
mode (or class for categorical variables) in that variable using frequency as proxy and sample a mode based on the logarithm of
its probability. Using the log probability instead of the original
frequency gives minority modes/classes higher chances to appear
during training. This helps to alleviate the collapse issue for rare
modes/classes.


**3.5** **Treat long tail**


We encode continuous values using variational Gaussian mixtures
to treat multi-mode data distributions (details in Sec. 3.3). However,
Gaussian mixtures can not deal with all types of data distribution,
notably distributions with long tail where few rare points are far
from the bulk of the data. VGM has difficulty to encode the values
towards the tail. To counter this issue we pre-process variables with
long tail distributions with a logarithm transformation. For such a
variable having values with lower bound _𝑙_, we replace each value _𝜏_
with compressed _𝜏_ _[𝑐]_ :


|Dataset|Train/Test<br>Split|Target<br>variable|#C|#B|#M|#Mi|
|---|---|---|---|---|---|---|
|Adult|39k/9k|"income"|3|2|7|2|
|Covertype|45k/5k|"Cover_Type"|9|44|1|1|
|Credit|40k/10k|"Class"|30|1|0|0|
|Intrusion|45k/5k|"Class"|4|6|14|18|
|Loan|4k/1k|"Personal-<br>Loan"|5|5|2|1|



_𝜏_ _[𝑐]_ = log( _𝜏_ ) if _𝑙_ _>_ 0
� log( _𝜏_   - _𝑙_ + _𝜖_ ) if _𝑙_ ⩽0, where _𝜖_ _>_ 0



(4)
�



state-of-the-art GAN based tabular data generators. We evaluate
the effectiveness of CTAB-GAN in terms of the resulting ML utility,
statistical similarity to the real data, and privacy distance. Moreover,
we provide an ablation analysis to highlight the efficacy of the
unique components of CTAB-GAN.


**4.1** **Experimental setup**


**Datasets** . Our algorithm is tested on five commonly used machine
learning datasets. Three of them – **Adult**, **Covertype** and **Intru-**
**sion** – are from the UCI machine learning repository [ 5 ]. The other
two – **Credit** [ 22 ] and **Loan** [ 9 ] – are from Kaggle. All five tabular datasets have a target variable, for which we use the rest of
the variables to perform classification. Due to computing resource
limitations, 50K rows of data are sampled randomly in a stratified
manner with respect to the target variable for Covertype, Credit
and Intrusion datasets. However, the Adult and Loan datasets are
not sampled. The details of each dataset are shown in Tab. 1.
**Baselines** . Our CTAB-GAN is compared with 4 state-of-the-art
GAN-based tabular data generators: CTGAN, TableGAN, CWGAN
and MedGAN. To have a fair comparison, all algorithms are coded
using Pytorch, with the generator and discriminator structures
matching the descriptions provided in their respective papers. For
Gaussian mixture estimation of continuous variables, we use the
same settings as the evaluation of CTGAN, i.e. 10 modes. All algorithms are trained for 150 epochs for Adult, Covertype, Credit
and Intrusion datasets, whereas the algorithms are trained for 300
epochs on Loan dataset. This is because, the Loan dataset is significantly smaller than the others containing only 5000 rows and
requires a long training time to converge. Lastly, each experiment
is repeated 3 times.
**Environment** . Experiments are run under Ubuntu 20.04 on a
machine equipped with 32 GB memory, a GeForce RTX 2080 Ti
GPU and a 10-core Intel i9 CPU.


**4.2** **Evaluation metrics**


The evaluation is conducted on three dimensions: (1) machine learning (ML) utility, (2) statistical similarity and (3) privacy preservability. The first two are used to evaluate if the synthetic data can be
used as a good proxy of the original data. The third criterion sheds
light on the nearest neighbour distances within and between the
original and synthetic datasets, respectively.



The log-transform allows to compress and reduce the distance
between the tail and bulk data making it easier for VGM to encode
all values, including tail ones. We show the effectiveness of this
simple yet performant method in Sec. 4.5.


**4** **EXPERIMENTAL ANALYSIS**


To show the efficacy of the proposed CTAB-GAN, we select five
commonly used machine learning datasets, and compare with four


KDD, 2021, Singapore Zilong Zhao, Aditya Kunar, Hiek Van der Scheer, Robert Birke, and Lydia Y. Chen


**Figure 6: Illustration of NNDR metric with its privacy risk**
**implications**



**Figure 5: Evaluation flows for ML utility**


_4.2.1_ _Machine learning (ML) utility._ As shown in Fig. 5, to evaluate
the ML utility of synthetic data, the original and synthetic data
are evaluated by 5 widely used machine learning algorithms: decision tree classifier, linear support-vector-machine (SVM), random
forest classifier, multinomial logistic regression and multi-layerperceptron (MLP).
We first split the original dataset into training and test datasets.
The training set is used as input to the GAN models as the real
dataset. Once the training is finished, we use the GAN models to
generate synthetic data with the same size as the training set. The
synthetic and real training datasets are then separately used to
train the above-mentioned machine learning models and evaluated
on the real test datasets. The machine learning performance is
measured via the accuracy, F1-score and area under the ROC. The
aim of this design is to test how close the ML utility is when we
train a machine learning model using the synthetic data vs the real
data. This responds to the question, “Can synthetic data be used as
a proxy of the original data for training ML models?”.


_4.2.2_ _Statistical Similarity._ Three metrics are used to quantitatively
measure the statistical similarity between the real and synthetic
data.

**Jensen-Shannon divergence (JSD)** [ 13 ]. The JSD provides a
measure to quantify the difference between the probability mass
distributions of individual categorical variables belonging to the
real and synthetic datasets, respectively. Moreover, this metric is
bounded between 0 and 1 and is symmetric allowing for an easy
interpretation of results.
**Wasserstein distance (WD)** [ 20 ]. In similar vein, the Wasserstein distance is used to capture how well the distributions of individual continuous/mixed variables are emulated by synthetically
produced datasets in correspondence to real datasets. We use WD
because we found that the JSD metric was numerically unstable
for evaluating the quality of continuous variables, especially when
there is no overlap between the synthetic and original dataset.
Hence, we resorted to utilize the more stable Wasserstein distance.
**Difference in pair-wise correlation (Diff. Corr.)** . To evaluate how well feature interactions are preserved in the synthetic
datasets, we first compute the pair-wise correlation matrix for the
columns within real and synthetic datasets individually. To measure



the correlation between any two continuous features, the Pearson
correlation coefficient is used. It ranges between [− 1 _,_ + 1 ] . Similarly,
the Theil uncertainty coefficient is used to measure the correlation between any two categorical features. It ranges between [ 0 _,_ 1 ] .
Lastly, the correlation ratio between categorical and continuous
variables is used. It also ranges between [ 0 _,_ 1 ] . Note that the dython [1]

library is used to compute these metrics. Finally, the differences between the pair-wise correlation matrices for the real and synthetic
datasets is computed.


_4.2.3_ _Privacy preservability._ To quantify the privacy preservability,
we resort to distance metrics (instead of differential privacy [ 27 ])
as they are intuitive and easy to understand by data science practitioners. Specifically, the following two metrics are used to evaluate
the privacy risk associated with synthetic datasets.
**Distance to Closest Record (DCR)** . The DCR is used to measure the Euclidean distance between any synthetic record and its
closest corresponding real neighbour. Ideally, the higher the DCR
the lesser the risk of privacy breach. Furthermore, the 5 _[𝑡ℎ]_ percentile of this metric is computed to provide a robust estimate of
the privacy risk.
**Nearest Neighbour Distance Ratio (NNDR)** [ 15 ]. Instead of
only measuring the closest neighbour, the NNDR measures the
ratio between the Euclidean distance for the closest and second

closest real neighbour to any corresponding synthetic record. This
ratio is within [ 0 _,_ 1 ] . Higher values indicate better privacy. Low
NNDR values between synthetic and real data may reveal sensitive
information from the closest real data record. Fig. 6 illustrates the
case. Hence, this ratio helps to evaluate the privacy risk with greater
depth and better certainty. Note that the 5 _[𝑡ℎ]_ percentile is computed
here as well.


**4.3** **Results analysis**


**ML Utility** . Tab. 2 shows the averaged ML utility differences between real and synthetic data in terms of accuracy, F1 score, and
AUC. A better synthetic data is expected to have low differences. It
can be seen that CTAB-GAN outperforms all other state-of-the-art
methods in terms of Accuracy, F1-score and AUC. Accuracy is the
most commonly used classification metric, but since we have imbalanced target variable, F1-score and AUC are more stable metrics
for such cases. AUC ranges value from 0 to 1. CTAB-GAN largely


1 [http://shakedzy.xyz/dython/modules/nominal/#compute_associations](http://shakedzy.xyz/dython/modules/nominal/#compute_associations)


CTAB-GAN: Effective Table Data Synthesizing KDD, 2021, Singapore


**(a) Covertype** **(b) Intrusion** **(c) Loan**


**Figure 7: ML utilities difference (i.e., AUC and F1-scoree) for five algorithms based on five synthetically generated data**



**Table 2: Difference of ML accuracy (%), F1-score, and AUC**
**between original and synthetic data: average over 5 different**
**datasets and 3 replications.**

|Method|Accuracy|F1-score|AUC|
|---|---|---|---|
|CTAB-GAN<br>CTGAN<br>TableGAN<br>MedGAN<br>CW-GAN|**9.83%**<br>21.51%<br>11.40%<br>14.11%<br>20.06%|**0.127**<br>0.274<br>0.130<br>0.282<br>0.354|**0.117**<br>0.253<br>0.169<br>0.285<br>0.299|



shortens the AUC difference from 0.169 (best in state-of-the-art) to

0.117.

To obtain a better understanding, Fig. 7 plots the (F1-score, AUC)
for all 5 ML models for the Covertype, Intrusion and Loan datasets.
Due to the page limit restrictions, results for Adult and Credit
datasets are not shown. Their results are similar as the ones of the

Covertype dataset. Furthermore, Fig. 7b shows that for the Intrusion
dataset CTAB-GAN largely outperforms all others across all ML
models used for evaluation. For datasets such as Covertype, the
results of CTAB-GAN and TableGAN are similar and clearly better
than the rest. This is because apart from CTGAN, the other models
fail to deal with the imbalanced categorical variables. Furthermore,
as CTGAN uses a VGM model with 10 modes, it fails to converge
to a suitable optimum for Covertype that mostly comprises single
mode Gaussian distributions.

For the Loan dataset, TableGAN is better than CTAB-GAN and
others, but the difference between the two is smaller than for the Intrusion dataset. We believe that the reason CTAB-GAN outperforms
the others by such a wide margin (17% higher than second best
for averaged accuracy across the 5 machine learning algorithms)
for the Intrusion dataset is that it contains many highly imbalanced categorical variables. In addition, Intrusion also includes 3
long tail continuous variables. Our results indicate that none of
the state-of-the-art techniques can perform well under these conditions. The Loan dataset is significantly smaller than the other 4
datasets and has the least number of variables. Moreover, all continuous variables are either simple one mode Gaussian distributions
or just uniform distributions. Therefore, we find that the encoding
method in CTAB-GAN which works well for complex cases, fails
to converge to a better optimum for simple and small datasets.



**Table 3: Statistical similarity: three measures averaged over**
**5 datasets and three repetitions.**

|Method|Avg JSD|Avg WD|Diff. Corr.|
|---|---|---|---|
|CTAB-GAN<br>CTGAN<br>TableGAN<br>MedGAN<br>CW-GAN|**0.0697**<br>0.0704<br>0.0796<br>0.2135<br>0.1318|**1050**<br>1769<br>2117<br>46257<br>238155|**2.10**<br>2.73<br>2.30<br>5.48<br>5.82|



**Statistical similarity** . Statistical similarity results are reported
in Tab. 3. CTAB-GAN stands out again across all comparisons. For
categorical variables (i.e. average JSD), CTAB-GAN and CTGAN
perform similarly (1% difference), and better than the other methods
by at least 12.4%, i.e. against the next best TableGAN. This is due
to the use of a conditional vector and the log-frequency sampling
of the training data, which works well for both balanced and imbalanced distributions. For continuous variables (i.e. average WD),
we still benefit from the design of the conditional vector. The average WD column shows some extreme numbers such as 46257 and
238155 comparing to 1050 of CTAB-GAN. The reason is that these
algorithms generate extremely large values for long tail variables.
Besides divergence and distance, our synthetic data also maintains
better correlation. We can see that TableGAN also performs well
here. However, as the extended conditional vector enhances the
training procedure, this helps to maintain even more so the correlation between variables. This is because the extended conditional

vector allows the generator to produce samples conditioned even
on a given VGM mode for continuous variables. This increases
the capacity to learn the conditional distribution for continuous
variables and hence leads to an improvement in the overall feature
interactions captured by the model.
**Privacy preservability** . As only PATE-GAN can generate synthetic data within tight differential privacy guarantees, we only
use distance-based algorithms to give an overview on privacy in
our evaluation. On the one hand, if the distance between real and
synthetic data is too large, it simply means that the quality of generated data is poor. On the other hand, if the distance between real
and synthetic data is too small, it simply means that there is a risk
to reveal sensitive information from the training data. Therefore,
the evaluation of privacy is relative. The privacy results are shown
in Tab. 4. It can be seen that the DCR and NNDR between real and


KDD, 2021, Singapore Zilong Zhao, Aditya Kunar, Hiek Van der Scheer, Robert Birke, and Lydia Y. Chen



**Table 4: Privacy impact: between real and synthetic data**
**(R&S) and within real data (R) and synthetic data (S).**

|Model|DCR|Col3|Col4|NNDR|Col6|Col7|
|---|---|---|---|---|---|---|
|**Model**|**R&S**|**R**|**S**|**R&S**|**R**|**S**|
|CTAB-GAN<br>CTGAN<br>TableGAN<br>MedGAN<br>CW-GAN|1.101<br>1.517<br>0.988<br>1.918<br>2.197|0.428<br>0.428<br>0.428<br>0.428<br>0.428|0.877<br>1.026<br>0.920<br>0.254<br>1.124|0.714<br>0.763<br>0.681<br>0.871<br>0.847|0.414<br>0.414<br>0.414<br>0.414<br>0.414|0.558<br>0.624<br>0.632<br>0.393<br>0.675|



synthetic data all indicate that generation from TableGAN has the
shortest distance to real data (highest privacy risk). The algorithm
which allows for greater distances between real and synthetic data
under equivalent ML utility and statistical similarity data should be
considered. In that case, CTAB-GAN not only outperforms TableGAN in ML utility and statistic similarity, but also in all privacy
preservability metrics by 10.3% and 4.6% for DCR and NNDR, respectively. Another insight from this table is that for MedGAN,
DCR within synthetic data is 41% smaller than within real data.
This suggests that it suffers from the mode collapse problem.


**4.4** **Ablation analysis**


To illustrate the efficiency of each strategy we implement an ablation study which cuts off the different components of CTAB-GAN
one by one:


**w/o classifier** . In this experiment, Classifier and the corresponding classification loss for Generator is taken away
from CTAB-GAN.

**w/o information loss** . In this experiment, we remove information loss from CTAB-GAN.

**w/o VGM and mode vector** . In this case, we substitute

VGM for continuous variables with min-max normalization

and use simple one-hot encoding for categorical variables.
Here the conditional vector is the same as for CTGAN.


The results are compared with the baseline implementing all
strategies. All experiments are repeated 3 times, and results are
evaluated on the same 5 machine learning algorithms introduced
in Sec. 4.2.1. The test datasets and evaluation flow are the same
as shown in Sec. 4.1 and Sec. 4.2. Tab. 5 shows the results. Each

part of CTAB-GAN has different impacts on different datasets. For
instance, **w/o classifier** has a negative impact for all datasets except Credit. Since Credit has only 30 continuous variables and one
target variable, the semantic check can not be very effective. **w/o**
**information loss** has a positive impact for Loan, but results degenerate for all other datasets. It can even make the model unusable,
e.g. for Intrusion. **w/o VGM and mode vector** performs bad for
Covertype, but has little impact for Intrusion. Credit w/o VGM
and mode vector performs better than original CTAB-GAN. This is
because out of 30 continuous variables, 28 are nearly single mode
Gaussian distributed. The initialized high number of modes, i.e. 10,
for each continuous variable (same setting as in CTGAN) degrades
the estimation quality. In general, if we average the column values,
all the ablation tests have a negative impact for the performance
which justifies our design choices for CTAB-GAN.



**Table 5: F1-score difference to CTAB-GAN. CTAB-GAN col-**
**umn reports the absolute averaged F1-score as baseline.**



|Dataset|w/o<br>Classifier|w/o<br>Info.<br>Loss|w/o VGM<br>and Mode<br>vector|CTAB-<br>GAN|
|---|---|---|---|---|
|Adult|-0.01|-0.037|-0.05|0.704|
|Covertype|-0.018|-0.184|-0.118|0.532|
|Credit|+0.011|-0.177|+0.06|0.71|
|Intrusion|-0.031|-0.437|+0.003|0.842|
|Loan|-0.044|+0.028|+0.013|0.803|


**4.5** **Further discussion**


After reviewing all the metrics, let us recall the three motivation
cases from Sec. 1.1.

**Mixed data type variables** . Fig. 8a compares the real and CTABGAN generated data for variable _Mortgage_ in the Loan dataset.
CTAB-GAN encodes this variable as mixed type. We can see that
CTAB-GAN generates clear 0 values. One drawback which can be
observed is that CTAB-GAN amplifies the dominance of the 0 in this
variable. Even with our log-frequency-based sampling of Gaussian
mixture modes and categorical classes, CTAB-GAN generates more
0 values than in the original distribution. That means there is still
room for improvement for extremely imbalanced cases.
**Long tail distributions.** Fig. 8b compares the cumulative frequency graph for the _Amount_ variable in Credit. This variable is
a typical long tail distribution. One can see that CTAB-GAN perfectly recovers the real distribution. Due to log-transform data preprocesssing, CTAB-GAN learns this structure significantly better
than the state-of-the-art methods shown in Fig. 1b.
**Skewed multi-mode continuous variables** . Fig. 8c compares
the frequency distribution for the continuous variable _Hours-per-_
_week_ from Adult. Except the dominant peak at 40, there are many
side peaks. Fig. 1c, shows that TableGAN, CWGAN and MedGAN
struggle since they can learn only a simple Gaussian distribution
due to the lack of any special treatment for continuous variables.
CTGAN, which also use VGM, can detect other modes. Still, CTGAN
is not as good as CTAB-GAN. The reason is that CTGAN lacks the
mode of continuous variables in the conditional vector. By incorporating the mode of continuous variables into conditional vector,
we can apply the training-by-sample and logarithm frequency also
to modes. This gives the mode with less weight more chance to
appear in the training and avoids the mode collapse.


**5** **CONCLUSION**


Motivated by the importance of data sharing and fulfillment of
governmental regulations, we propose CTAB-GAN– a conditional
GAN based tabular data generator. CTAB-GAN advances beyond
the prior state-of-the-art methods by modeling mixed variables and
provides strong generation capability for imbalanced categorical
variables, and continuous variables with complex distributions. To
such ends, the core features of CTAB-GAN include (i) introduction
of the classifier into conditional GAN, (ii) effective data encoding for
mixed variable, and (iii) a novel construction of conditional vectors.
We exhaustively evaluate CTAB-GAN against four tabular data
generators on a wide range of metrics, namely resulting ML utilities,










CTAB-GAN: Effective Table Data Synthesizing KDD, 2021, Singapore


**(a) Mortgage in Loan** **(b) Amount in Credit** **(c) Hours-per-week in Adult**


**Figure 8: Challenges of modeling industrial dataset using existing GAN-based table generator: (a) mixed type, (b) long tail**
**distribution, and (c) skewed data**



statistical similarity and privacy preservation. The results show
that the synthetic data of CTAB-GAN results into high utilities,
high similarity and reasonable privacy guarantee, compared to
existing state-of-the-art techniques. The improvement on complex
datasets is up to 17% in accuracy comparing to all state-of-the-art
algorithms. The remarkable results of CTAB-GAN demonstrate
its potential for a wide range of applications that greatly benefit
from data sharing, such as banking, insurance, manufacturing, and
telecommunications.


**ACKNOWLEDGEMENTS**


This work has been funded by Ageon data science division.


**REFERENCES**


[1] M. Arjovsky, S. Chintala, and L. Bottou. Wasserstein generative adversarial
networks. In _Proceedings of the 34th International Conference on Machine Learning_

_- Volume 70_, page 214–223. JMLR.org, 2017.

[2] M. G. Bellemare, I. Danihelka, W. Dabney, S. Mohamed, B. Lakshminarayanan,
S. Hoyer, and R. Munos. The cramer distance as a solution to biased wasserstein
gradients. _ArXiv_, abs/1705.10743, 2017.

[3] C. M. Bishop. _Pattern Recognition and Machine Learning (Information Science and_
_Statistics)_ . Springer-Verlag, Berlin, Heidelberg, 2006.

[4] E. Choi, S. Biswal, B. Malin, J. Duke, W. F. Stewart, and J. Sun. Generating
multi-label discrete patient records using generative adversarial networks. _arXiv_
_preprint arXiv:1703.06490_, 2017.

[5] [D. Dua and C. Graff. UCI machine learning repository. http://archive.ics.uci.edu/](http://archive.ics.uci.edu/ml)
[ml, 2017.](http://archive.ics.uci.edu/ml)

[6] J. Engelmann and S. Lessmann. Conditional wasserstein gan-based oversampling
of tabular data for imbalanced learning. _arXiv preprint arXiv:2008.09202_, 2020.

[7] I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair,
A. Courville, and Y. Bengio. Generative adversarial nets. In _Proceedings of the_
_27th International Conference on Neural Information Processing Systems - Volume_
_2_, page 2672–2680, Cambridge, MA, USA, 2014.

[8] I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, and A. Courville. Improved
training of wasserstein gans. In _Proceedings of the 31st International Conference_
_on Neural Information Processing Systems_, page 5769–5779, Red Hook, NY, USA,
2017.

[9] [S. Jacob. Kaggle - personal loan classification problem. https://www.kaggle.com/](https://www.kaggle.com/itsmesunil/bank-loan-modelling)
[itsmesunil/bank-loan-modelling, 2019.](https://www.kaggle.com/itsmesunil/bank-loan-modelling)

[10] T. Karras, S. Laine, and T. Aila. A style-based generator architecture for generative
adversarial networks. In _IEEE/CVF Conference on Computer Vision and Pattern_
_Recognition (CVPR)_, pages 4396–4405, 2019.




[11] T. Karras, S. Laine, M. Aittala, J. Hellsten, J. Lehtinen, and T. Aila. Analyzing and
improving the image quality of stylegan. In _IEEE/CVF Conference on Computer_
_Vision and Pattern Recognition (CVPR)_, pages 8107–8116, 2020.

[12] Y. LeCun, Y. Bengio, and G. Hinton. Deep learning. _Nature_, 521(7553):436–444,
2015.

[13] J. Lin. Divergence measures based on the shannon entropy. _IEEE Transactions on_
_Information Theory_, 37(1):145–151, 1991.

[14] Z. Lin, A. Khetan, G. Fanti, and S. Oh. Pacgan: The power of two samples in
generative adversarial networks. _IEEE Journal on Selected Areas in Information_
_Theory_, 1(1):324–335, 2020.

[15] D. G. Lowe. Distinctive image features from scale-invariant keypoints. _Int. J._
_Comput. Vision_, 60(2):91–110, Nov. 2004.

[16] A. Mottini, A. Lheritier, and R. Acuna-Agost. Airline Passenger Name Record
Generation using Generative Adversarial Networks. In _workshop on Theoretical_
_Foundations and Applications of Deep Generative Models. ICML_, July 2018.

[17] A. Narayanan and V. Shmatikov. Robust de-anonymization of large sparse
datasets. In _IEEE Symposium on Security and Privacy_, pages 111–125, 2008.

[18] A. Odena, C. Olah, and J. Shlens. Conditional image synthesis with auxiliary
classifier gans. In _Proceedings of the 34th International Conference on Machine_
_Learning - Volume 70_, page 2642–2651. JMLR.org, 2017.

[19] N. Park, M. Mohammadi, K. Gorde, S. Jajodia, H. Park, and Y. Kim. Data synthesis
based on generative adversarial networks. _Proc. VLDB Endow._, 11(10):1071–1083,
June 2018.

[20] A. Ramdas, N. G. Trillos, and M. Cuturi. On wasserstein two-sample testing and
related families of nonparametric tests. _Entropy_, 19(2), 2017.

[21] The Economist. The world’s most valuable resource is no longer oil, but
[data. https://www.economist.com/leaders/2017/05/06/the-worlds-most-valuable-](https://www.economist.com/leaders/2017/05/06/the-worlds-most-valuable-resource-is-no-longer-oil-but-data)
[resource-is-no-longer-oil-but-data, 2017.](https://www.economist.com/leaders/2017/05/06/the-worlds-most-valuable-resource-is-no-longer-oil-but-data)

[22] M. L. G. ULB. Kaggle - anonymized credit card transactions labeled as fraudulent
[or genuine. https://www.kaggle.com/mlg-ulb/creditcardfraud, 2018.](https://www.kaggle.com/mlg-ulb/creditcardfraud)

[23] R. Wang, B. Fu, G. Fu, and M. Wang. Deep & cross network for ad click predictions.
In _Proceedings of the ADKDD’17_, New York, NY, USA, 2017.

[24] L. Xu, M. Skoularidou, A. Cuesta-Infante, and K. Veeramachaneni. Modeling
tabular data using conditional gan. In _Advances in Neural Information Processing_
_Systems, 2019_, volume 32, pages 7335–7345. Curran Associates, Inc., 2019.

[25] L. Xu and K. Veeramachaneni. Synthesizing tabular data using generative adversarial networks. _arXiv preprint arXiv:1811.11264_, 2018.

[26] A. Yahi, R. Vanguri, and N. Elhadad. Generative adversarial networks for electronic health records: A framework for exploring and evaluating methods for
predicting drug-induced laboratory test trajectories. In _Neural Information Pro-_
_cessing Systems (NIPS) workshop_, 2017.

[27] J. Yoon, J. Jordon, and M. van der Schaar. PATE-GAN: Generating synthetic data
with differential privacy guarantees. In _International Conference on Learning_
_Representations_, 2019.


KDD, 2021, Singapore Zilong Zhao, Aditya Kunar, Hiek Van der Scheer, Robert Birke, and Lydia Y. Chen



**A** **USER MANUAL**

**A.1** **Introduction**


The software demo developed by us comprises of a synthetic tabular
data generation pipeline. It was implemented using python 3.7.*
along with the flask library to work as a web application on a local
server. The application functionality and usage can be found listed
under the Functionality & Usage sections respectively. In addition,
[a video of the demo can be seen here.](https://drive.google.com/file/d/1VK6479YPnjg0zVbfdgJb2G4_7lz2CWp6/view)


**A.2** **Functionality**


Our demo comprises of the following salient features:


(1) **Synthetic Data Generator:** Our software is a cross-platform
application that sits on top of a python interpreter. Moreover,
it is relatively lightweight and can be set-up easily using pip.
Our application is also robust against missing values and supports date-type formats. We believe these factors increases
its usability in real-world scenarios.
(2) **Synthetic Data Evaluator:** In addition to our generator,
we also provide a detailed evaluation of the synthetic data.
The report provides end users with visual plots comparing
the real and synthetic distributions of individual columns
as shown in sub-figures A.1a & A.1b of Fig. A.1. In addition,
the synthetic data’s utility for ML applications along with
its privacy preservability metrics are reported as can be seen
in sub-figures A.2a & A.2b of Fig. A.2. Note that the tableevaluator [2] library aided us in generating this evaluating
report.


**A.3** **Usage**


The following step-by-step instructions are provided to allow endusers to use our product in a hassle-free manner.


**Step 1:** Open the terminal and navigate to the root directory
of the software package to run the following command
python3 / python server.py.
**Step 2:**


Open the browser, the application should now be
available at the following address:
http://127.0.0.1:5000/.
**Step 3:** If this is the first time running the web application, it is
advised to click on the “Train a new model” button to

begin training the model with a dataset. Otherwise, click
on the “Use existing mode” button to use an existing
trained model. If you clicked on the “Use existing mode”
button, please go to **step 8** . If not, please continue with
the next step.


2 [ttps://github.com/Baukebrenninkmeijer/Table-Evaluator](ttps://github.com/Baukebrenninkmeijer/Table-Evaluator)



**Step 4:**


Click on the “Browse” button to select the dataset for

which the model needs to train. Afterwards click on the
“Uploap” button.
**Step 5:**


The software will auto-detect the column types, and
will give the option to adjust a column’s data type and
inclusion in the training. Note that the red highlighted
column shows the current columns in the uploaded csv
file. The yellow highlighted column gives the option to
include or exclude a particular column in the training
process by clicking on the switch button. The
highlighted green column is the auto detected data
type. It also has the option to be adjusted as needed.
Simply click on it and select the desired data type from
the drop down menu. Click on the “Submit” button
after choosing the right settings.
**Step 6:**


In the following page, specify the problem type for the
given dataset. The software currently provides the
following problem types: None, Binary Classification
and Multi-class Classification. If unsure, leave it as
None. Then enter the number of epochs needed to train
the model. Click on “Train Model” to start the training.


CTAB-GAN: Effective Table Data Synthesizing KDD, 2021, Singapore



**Step 7:**


Once the model has finished training, the option to
train a new model or proceed to the synthesizer is
presented. To generate synthetic data, click on
“Proceed to data synthesizer”.
**Step 8:**


The trained models can be found in the dropdown
menu of the Models field, as seen in the figure above.
Click on it, and select the trained model. After this, type
the amount of rows to be generated in the second field,
and click on “Start Synthesizer” to start the process.
**Step 9:**


Once the data is generated, the following page shows a
snippet of the synthetic data. The generated data can
be saved locally by clicking on “Download csv”. This
page also gives you the option to generate a report for
the given data. In order to generate the report in PDF
format, simply click on “Generate report” and continue
with step 10.
**Step 10:**


The following page is presented while the report is
being generated. It will automatically redirect to the
PDF once it is completed.



**Figure A.1: Visual plots comparing the generated vs real data**
**distribution.**


**(a) ML Utility**


**(b) Privacy Preservability**


**Figure A.2: ML utility and privacy preservability of the gen-**
**erated data.**



**Step 11:**



Finally, once the PDF has been generated, it can be
saved locally by clicking on “Save as” or “Print” in the
browser.



**(a) Cumulative distribution compari-**
**son of Age in Adult**



**(b) Frequency comparison of cate-**
**gories within Workclass in Adult**


