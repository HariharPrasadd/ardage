## **Homomorphic Encryption and Federated Learning based** **Privacy-Preserving CNN Training: COVID-19 Detection** **Use-Case**



Febrianti Wibawa [∗]

Ferhat Ozgur Catak [†]

f.febrianti@gmail.com
f.ozgur.catak@uis.no
Department of Electrical Engineering and Computer
Science, University of Stavanger
Stavanger, Rogaland, Norway


Murat Kuzlu

mkuzlu@odu.edu
Batten College of Engineering and Technology, Old
Dominion University
Norfolk, Virginia, USA


**ABSTRACT**


Medical data is often highly sensitive in terms of data privacy and
security concerns. Federated learning, one type of machine learning techniques, has been started to use for the improvement of
the privacy and security of medical data. In the federated learning,
the training data is distributed across multiple machines, and the
learning process is performed in a collaborative manner. There
are several privacy attacks on deep learning (DL) models to get
the sensitive information by attackers. Therefore, the DL model
itself should be protected from the adversarial attack, especially for
applications using medical data. One of the solutions for this problem is homomorphic encryption-based model protection from the
adversary collaborator. This paper proposes a privacy-preserving
federated learning algorithm for medical data using homomorphic encryption. The proposed algorithm uses a secure multi-party
computation protocol to protect the deep learning model from the
adversaries. In this study, the proposed algorithm using a real-world
medical dataset is evaluated in terms of the model performance.


**CCS CONCEPTS**


- **Theory of computation** → **Cryptographic protocols** ; - **Secu-**
**rity and privacy** ;


**KEYWORDS**


Homomorphic encryption, sensitive health data, federated learning,
secure multi-party computation


Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than ACM
must be honored. Abstracting with credit is permitted. To copy otherwise, or republish,
to post on servers or to redistribute to lists, requires prior specific permission and/or a
fee. Request permissions from permissions@acm.org.
_EICC ’22, June 15–16, 2022, Barcelona, Spain_
© 2022 Association for Computing Machinery.
ACM ISBN 978-1-4503-XXXX-X/18/06...$15.00
[https://doi.org/XXXXXXX.XXXXXXX](https://doi.org/XXXXXXX.XXXXXXX)



Salih Sarp
sarps@vcu.edu
Department of Electrical and Computer Engineering,
Virginia Commonwealth University
Richmond, Virginia, USA


Umit Cali

umit.cali@ntnu.no
Norwegian University of Science and Technology
Trondheim, Norway


**ACM Reference Format:**

Febrianti Wibawa, Ferhat Ozgur Catak, Salih Sarp, Murat Kuzlu, and Umit
Cali. 2022. Homomorphic Encryption and Federated Learning based PrivacyPreserving CNN Training: COVID-19 Detection Use-Case. In _European_
_Interdisciplinary Cybersecurity Conference, June 15–16, 2022, Barcelona, Spain._
[ACM, New York, NY, USA, 6 pages. https://doi.org/XXXXXXX.XXXXXXX](https://doi.org/XXXXXXX.XXXXXXX)


**1** **INTRODUCTION**


Machine learning (ML) is a widely used technique in almost all
fields, where a computer system can learn from data to improve
its performance. This technique is widely used in many application areas such as image recognition, natural language processing,
and machine translation. Federated learning is a machine learning
technique where the training data is distributed across multiple
machines, and the learning process is performed in a collaborative
manner [ 13 ]. This technique can be used to improve the privacy
and security of medical data [10].
Medical data is often highly sensitive and is often subject to data
privacy and security concerns [ 1 ]. For example, a person’s health
information is often confidential and can be used to identify the
person. Thus it is essential to protect the privacy and security of
medical data. Health Insurance Portability and Accountability Act
(HIPAA) (US Department of Health and Human Services, 2014) and
General Data Protection Regulation (GDPR) (The European Union
,2018) strictly mandate the personal health information privacy.
There are various methods to safeguard the private information.
Federated learning is one of the techniques that can be utilized for
the protection of sensitive data during multi-party computation
tasks. This technique can be used to improve the privacy and security of medical data by preventing the data from being centralized
and vulnerable.

Keeping the data local is not sufficient for the security of the
data and the ML model. However, there are several privacy attacks
on deep learning models to get the private data [ 9, 25 ]. For example,
the attackers can use the gradient information of the deep learning
model to get the sensitive information. Thus the deep learning


EICC ’22, June 15–16, 2022, Barcelona, Spain Wibawa and Catak, et al.



model itself should be protected from the adversaries as well. One
of the solutions for this problem is homomorphic encryption-based
model protection from the adversary collaborator. Homomorphic
encryption is a technique where the data can be encrypted, and
the operations can be performed on the encrypted data [ 4 ]. This
technique can be used to protect the deep learning model from the
adversaries.

This paper proposes a privacy-preserving federated learning
algorithm based convolutional neural network (CNN) for medical data using homomorphic encryption. The proposed algorithm
uses a secure multi-party computation protocol to protect the deep
learning model from the adversaries. We evaluate the proposed
algorithm using a real-world medical dataset and show that the
proposed algorithm can protect the deep learning model from the
adversaries.


**2** **RELATED WORK**


Data-driven ML models provide unprecedented opportunities for
healthcare with the use of sensitive health data. These models are

trained locally to protect the sensitive health data. However, it is
difficult to build robust models without diverse and large datasets
utilizing the full spectrum of health concerns. Prior proposed works
to overcome this problems include federated learning techniques.
For instance, the studies [ 5, 17, 24 ] reviewed the current applications
and technical considerations of the federated learning technique
to preserve the sensitive biomedical data. Impact of the federated
learning is examined through the stakeholders such as patients,
clinicians, healthcare facilities and manufacturers. In another study,
the authors in [ 16 ] utilized federated learning systems for brain tumour segmentation on the BraTS dataset which consist of magnetic
resonance imaging brain scans. The results show that performance
is decreased by the privacy protection costs. Same BraTS dataset
is used in [ 19 ] to compare three collaborative training techniques,
i.e., federated learning, institutional incremental learning (IIL) and
cyclic institutional learning (CIIL). In IIL and CIIL, institutions train
a shared model successively where CIIL adds a cycling loop through
organisations. The results indicates that federated learning achieves
similar Dice scores to that of models trained by sharing data. It
outperform the IIL and CIIL methods since these methods suffer
from catastrophic forgetting and complexity.
Medical data is also safeguarded by encryption techniques such
as homomorphic encryption. In [ 15 ], authors propose an online
secure multiparty computation with sharing patient information to
hospitals using homomorphic encryption. Bocu et al. [ 7 ] proposed
a homomorphic encryption model that is integrated to personal
health information system utilizing heart rate data. The results
indicates that the described technique successfully addressed the
requirements for the secure data processing for the 500 patients
with expected storage and network challenges. In another study
by Wang et al. [ 23 ] proposed a data division scheme based homomorphic encryption for wireless sensor networks. The results show
that there is trade off between resources and data security. In [ 14 ],
applicability of homomorphic encryption is shown by measuring
the vitals of the patients with a lightweight encryption scheme.
Sensor data such as respiration and heart rate are encrypted using
homomorphic encryption before transmitting to the non-trusting



third party while encryption takes place only in medical facility.
The study in [ 20 ] developed an IoT based architecture with homomorphic encryption to combat data loss and spoofing attacks
for chronic disease monitoring. results suggest that homomorphic
encryption provide cost effective and straightforward protection to
the sensitive health information. Blockchain technologies are also
utilized in cooperation with homomorphic encryption for the security of medical data. Authors in [ 21 ] proposed a practical pandemic
infection tracking using homomorphic encryption and blockchain
technologies in intelligent trasnportatiton systems using automatic
healthcare monitoring. In another study Ali et al. [ 3 ] developed
a search-able distributed medical database on a blockchain using
homomorphic encryption. The increase need to secure sensitive
information leads to use of various techniques together. In the
scope of this study, a multi-party computation tool using federated
learning with homomorphic encryption is developed and analyzed.


**3** **PRELIMINARIES**

**3.1** **Homomorphic encryption**


Nowadays data encryption is a common practice not only for enterprises but also individuals. It is meant to protect privacy of the data.
Data encryption mostly done at rest, when the data is stored and
in transit when the data is transferred. However data encryption is
not popularly used upon when running or executing the operations
or computations.
Homomorphic encryption is an encryption method which allows
arithmetical computations to be performed directly on encrypted
or ciphered text without requiring any decryption. Outputs of the
computations are also in encrypted form and provide identical or
almost identical result when decrypted. This means that Homomorphic encryption allows data processing without disclosing the
actual data.

If _𝐸𝑛𝑐_ denotes encryption, _𝐷𝑒𝑐_ denotes decryption, and _𝑓_ () is
a function applied on actual values (plaintexts) _𝑎_ and _𝑏_, using encrpytion key _𝑝𝑘_, then homomorphic encryption property would be:


_𝑓_ ( _𝑎,𝑏_ ) = _𝐷𝑒𝑐_ ( _𝐸𝑛𝑐_ ( _𝑝𝑘,𝑎_ ) _, 𝐸𝑛𝑐_ ( _𝑝𝑘,𝑏_ ))
Homomorphic encryption can be used for privacy-preserving
outsourced storage and computation. This allows data to be encrypted and out-sourced to commercial cloud environments for
processing, all while encrypted.
There are several types of homomorphic encryption [2];


(1) Partially homomorphic encryption is homomorphic encryption that supports only one homomorphic operation, either
addition or multiplication, with unlimited number of times.
(2) Somewhat homomorphic encryption schemes allows both
addition and multiplication but only in a limited number of

times.

(3) Leveled fully homomorphic encryption supports the evaluation of arbitrary circuits composed of multiple types of
gates of bounded (pre-determined) depth.
(4) Fully homomorphic encryption (FHE) supports both addition
and multiplication operations with unlimited number of

times.


Homomorphic Encryption and Federated Learning based Privacy-Preserving CNN Training: COVID-19 Detection Use-Case EICC ’22, June 15–16, 2022, Barcelona, Spain



Somewhat homomorphic encryption (SHE) is used in this work
since it allows both addition and multiplication operations on encrypted data which is required in aggregation of machine learning
model weights.


**3.2** **Brakerski-Fan-Vercauteren (BFV) scheme**


The BFV scheme is a well-known homomorphic encryption scheme.
It encrypts polynomials instead of bits. The encrypted polynomials
can be evaluated homomorphically. It is secure in the sense that it
is CCA secure. The security is based on the hardness of the problem
SIS. It can be described as follows.

We now briefly describe the BFV scheme. Let _𝑛_ be a positive
integer, _𝑞_ be a prime number, F _𝑞_ be the finite field with _𝑞_ elements,
_𝑡_ be a positive integer, ( _𝛼_ 0 _, 𝛼_ 1 _, . . ., 𝛼_ _𝑛_ −1 ) be a random tuple in F _𝑞_ _[𝑛]_,
_𝑠_ be a positive integer, _𝜂_ be a positive integer. Let _𝑁_ = _𝑞_ _[𝑠]_ and
_𝑀_ = _𝑞_ _[𝜂]_ . The secret key is ( _𝛼_ 0 _, 𝛼_ 1 _, . . ., 𝛼_ _𝑛_ −1 ) . The public key is
( _𝛼_ 0 _[𝑁]_ _[, 𝛼]_ 1 _[𝑁]_ _[, . . ., 𝛼]_ _𝑛_ _[𝑁]_ −1 [)] [. The message space is] [ F] _[𝑞]_ [[] _[𝑥]_ []] _[<][𝑡]_ [. The message]

_𝑚_ ( _𝑥_ )
_𝑚_ ( _𝑥_ ) is encrypted to _𝑐_ ( _𝑥_ ) = ( _𝑥_ − _𝛼_ 0 ) ( _𝑥_ − _𝛼_ 1 ) _..._ ( _𝑥_ − _𝛼_ _𝑛_ −1 ) [+] _[ 𝑒]_ [(] _[𝑥]_ [)] [, where]
_𝑒_ ( _𝑥_ ) is a polynomial of degree less than _𝑡_ . The decryption is done by
evaluating _𝑐_ ( _𝑥_ ) at all points of the form _𝛼_ _𝑖_ _[𝑀]_ and then interpolating
_𝑚_ ( _𝑥_ ) from the resulting evaluations.


**3.3** **Regulatory Aspects of Privacy in Health**
**Sector**


Trust and privacy are among the fundamental elements of digital healthcare systems and platforms. The trust is expected to be
built between various stakeholders of the digital healthcare ecosystems such as patients, medical care providers, health authorities
and healthcare systems providers. The following medical data are
among the most critical ones in terms of privacy and have to be
protected:


  - Personal information related to patient such as address, social security number, birth date, and bank account number,

  - Provided medical and psychological services, drugs, equipment, and procedures,

   - Status of the patients’ medical or psychological conditions,

  - The information related to the hospital, clinic or the medical
professionals who provided the medical and psychological

services.


The European General Data Protection Regulation (GDPR) is among
the mostly applied regulatory framework in terms of data privacy
that concentrates on individual control for data subjects of ‘their’
data. Public and private healthcare data privacy is handled under
GDPR regulations [22].


**3.4** **BFV Scheme**


The Brakerski/Fan-Vercauteren (BFV) architecture [ 8, 11, 12 ] incorporates powerful Single Instruction Multiple Data (SIMD) parallelism, making it ideal for applications that handle massive volumes
of data. In this crypto scheme, the messages are the vectors of integers, m ∈ Z _[𝑛]_ . The messages are encoded into plaintext polynomials
of degree _𝑛_ .



**3.5** **Federated Learning**


Federated learning is a machine learning technique that enables
multiple parties to build and train a common machine learning
model without exchanging or sharing data. Each party (client) stores
and processes their own dataset (local dataset) while there is a
common model shared with all parties (clients). In this case each
client trains the common model using local dataset, and sends
trained model to a centralized server. The server then aggregates
model received from all the clients and distributes the aggregated
model back to the clients.

Federated learning addresses data security and privacy issues
since it doesn’t require access to dataset of each client, nor requires
the dataset to be distributed. The local dataset itself doesn’t have to

be identically distributed and can be heterogeneous. This behaviour
makes Federated Learning more popular in healthcare applications.
Federated Learning enables health institutions to form and train a
common model without transferring sensitive patient data out.
There are several types of Federated Learning setting:[6]


(1) Centralized federated learning. In this setting, a central server
is used to populate and aggregate models from participating
clients during learning process. A global common model is
pushed from the server down to the clients.
(2) Decentralized federated learning. In this setting, participating clients coordinate among themselves to obtain a global
common model [18].
(3) Heterogeneous federated learning. In this setting, participating clients come from different technical platfrom, e.g. PC
and mobile phones, with own local dataset and model while
obtaining single global model.

In this work, centralized federated learning setting is implemented, to demonstrated model aggregation by single centralized

server.


**4** **SYSTEM MODEL**


This section gives a high-level system overview of the proposed
BFV crypto-scheme-based privacy-preserving federated learning
COVID-19 detection training method. The proposed privacy-preserving
scheme is a two-phase approach: (1) local model training at each
client and (2) encrypted model weight aggregation at the server.
In the local model training phase, each client builds their local
CNN based DL model using their local electronic health record
dataset. The clients encrypt the model weights matrix using the
public key. In the second step, the server aggregates all clients’
encrypted weight matrices and sends the final matrix to the clients.
Each client decrypts the aggregated encrypted weight matrix to
update the model weights of their DL model. Figure 1 shows the
system overview.
Figure 2 shows CNN based COVID-19 detection model used in
the experiments.


**4.1** **Notations**


   - Boldface lowercase letters show the vectors (e.g., x)

  - ⟦ _𝑊_ ⟧ shows the ciphertext of a matrix _𝑊_ .

  - ⊕ shows homomorphic encryption based addition, ⊗ homomorphic encryption based multiplication.

  - ( _𝑘𝑒𝑦_ _𝑝𝑢𝑏_ _,𝑘𝑒𝑦_ _𝑝𝑟𝑖𝑣_ ) shows public/private key pairs.


EICC ’22, June 15–16, 2022, Barcelona, Spain Wibawa and Catak, et al.


**Figure 1: Overall system overview of the proposed method**


**4.3** **Model Aggregation**

















64



**fc1**



**Figure 2: CNN based COVID-19 detection model.**


**4.2** **Client Initialization**


Algorithm 1 shows the overall process in the initialization phase.
Each client trains the local classifier, _ℎ_ _𝑖_ with their private datase,
D _𝑖_ . The trained model’s weight matrix, _𝑊_, is encrypted, ⟦ _𝑊_ ⟧, and
shared with the server


**Algorithm 1** Model training in each client


**Require:** The dataset at client _𝑐_ : D _𝑐_ = {(x _, 𝑦_ ) |x ∈ R _[𝑚]_ _, 𝑦_ ∈ R} _[𝑚]_ _𝑖_ =0 [, public]
key: _𝐾𝑒𝑦_ _𝑝𝑢𝑏_
1: _𝑋_ _𝑡𝑟𝑎𝑖𝑛_ _,𝑋_ _𝑡𝑒𝑠𝑡_ _,_ y _𝑡𝑟𝑎𝑖𝑛_ _,_ y _𝑡𝑒𝑠𝑡_ ← _𝑡𝑟𝑎𝑖𝑛_ _ _𝑡𝑒𝑠𝑡_ _ _𝑠𝑝𝑙𝑖𝑡_ (D)
2: _ℎ_ ← _𝑔𝑙𝑜𝑏𝑎𝑙_ _ _𝑚𝑜𝑑𝑒𝑙_
3: _ℎ.𝑓𝑖𝑡_ ( _𝑋_ _𝑡𝑟𝑎𝑖𝑛_ _,_ y _𝑡𝑟𝑎𝑖𝑛_ )
4: _𝑊_ ←∅ // _Create an empty matrix for the encrypted layer weights_
5: **for each** _𝑙𝑎𝑦𝑒𝑟_ ∈ _ℎ_ **do**
6: ⟦ _𝑊_ ⟧← _𝑒𝑛𝑐𝑟𝑦𝑝𝑡_ _ _𝑓𝑟𝑎𝑐𝑡𝑖𝑜𝑛𝑎𝑙_ ( _𝑙𝑎𝑦𝑒𝑟.𝑤𝑒𝑖𝑔ℎ𝑡𝑠,𝑘𝑒𝑦_ _𝑝𝑢𝑏_ ) // _En-_
_crypt the layer weights (𝑙𝑎𝑦𝑒𝑟.𝑤𝑒𝑖𝑔ℎ𝑡𝑠_ ∈ R _[𝑚]_ _) with public key._
7: **end for**
8: **Return** ⟦ _𝑊_ ⟧ // _The encrypted weight matrix_



The server collects all encrypted weight matrices, {⟦ _𝑊_ ⟧ 0 _,_ - · · _,_ ⟦ _𝑊_ ⟧ _𝑐_ },
from the clients. It calculates the average weight value of each neuron in the encrypted domain. Algorithm 2 shows the overall process
in the aggregation phase.


**Algorithm 2** Model aggregation at the server


**Require:** public key: _𝐾𝑒𝑦_ _𝑝𝑢𝑏_, the number of clients: _𝑐_, client model
weights: _𝐻_ = {⟦ _𝑊_ ⟧ _𝑖_ } _[𝑐]_ _𝑖_ =0
1: ⟦ _𝑊_ ⟧ _𝑎𝑔𝑔𝑟_ ←∅
2: **for each** _ℎ_ ∈ _𝐻_ **do**
3: **for each** ⟦ _𝑟𝑜𝑤_ ⟧∈ _ℎ_ **do**
4: ⟦ _𝑊_ ⟧ _𝑎𝑔𝑔𝑟_ ←⟦ _𝑊_ ⟧ _𝑎𝑔𝑔𝑟_ ⊕⟦ _𝑟𝑜𝑤_ ⟧ // _Homomorphic addition_
5: **end for**

6: **end for**
7: **for each** ⟦ _𝑟𝑜𝑤_ ⟧∈⟦ _𝑊_ ⟧ _𝑎𝑔𝑔𝑟_ **do**
8: ⟦ _𝑟𝑜𝑤_ ⟧←⟦ _𝑟𝑜𝑤_ ⟧⊗ _𝑐_ [−][1] // _Homomorphic multiplication._
9: **end for**
10: **Return** ⟦ _𝑊_ ⟧ _𝑎𝑔𝑔𝑟_ // _Return the aggregated weight matrix in the en-_
_crypted domain_


**4.4** **Client Decryption**


The last step is client decryption which each client decrypt the
aggregated and encrypted weight matrix, ⟦ _𝑊_ ⟧ _𝑎𝑔𝑔𝑟_, and updates
their local model, _ℎ_ . Algorithm 3 shows the overall process in the
client decryption phase.


**5** **RESULTS**

**5.1** **Experimental Setup**


We have implemented our proposed protocols and the classifier
training phase in Python by using the Keras/Tensorflow libraries
for the model building and the Microsoft SEAL library for the
somewhat homomorphic encryption implementation. To show the
training phase time performance of the proposed protocols, we
tested COVID-19 x-ray scans public dataset with different number


Homomorphic Encryption and Federated Learning based Privacy-Preserving CNN Training: COVID-19 Detection Use-Case EICC ’22, June 15–16, 2022, Barcelona, Spain



**Algorithm 3** Client decryption


**Require:** private key: _𝐾𝑒𝑦_ _𝑝𝑟𝑖𝑣_, encrypted aggregated weights: ⟦ _𝑊_ ⟧ _𝑎𝑔𝑔𝑟_

1: _ℎ_ ← _𝑔𝑙𝑜𝑏𝑎𝑙_ _ _𝑚𝑜𝑑𝑒𝑙_
2: **for each** _𝑙𝑎𝑦𝑒𝑟_ ∈ _ℎ_ **do**
3: ⟦ _𝑟𝑜𝑤_ ⟧←⟦ _𝑊_ ⟧ _𝑎𝑔𝑔𝑟_ ( _𝑙𝑎𝑦𝑒𝑟_ ) // _Get the corresponding row for layer_
4: _𝑙𝑎𝑦𝑒𝑟_ ← _𝑑𝑒𝑐𝑟𝑦𝑝𝑡_ _ _𝑓𝑟𝑎𝑐𝑡𝑖𝑜𝑛𝑎𝑙_ (⟦ _𝑟𝑜𝑤_ ⟧ _,𝑘𝑒𝑦_ _𝑝𝑟𝑖𝑣_ ) // _Decrypt the_
_row and update the layer weights_
5: **end for**

6: _ℎ.𝑠𝑎𝑣𝑒_ _ _𝑚𝑜𝑑𝑒𝑙_ ( _𝑔𝑙𝑜𝑏𝑎𝑙_ _ _𝑚𝑜𝑑𝑒𝑙_ ) // _Save the aggregated model as_
_global_model at client._


**(a)** **(b)** **(c)**


**(d)** **(e)** **(f)**


**Figure 3: An example of an X-ray scan images taken from**
**the dataset (a, b, c) with a label of COVID-19 negative, (d, e,**
**f) COVID-19 positive.**


of clients and the ciphertext modulus, _𝑞_ = { 128 _,_ 192 }, which determines how much noise can accumulate before decryption fails.
Table 1 shows the dataset details.


**Table 1: Dataset description**

|Dataset|Rows|Label|
|---|---|---|
|**Training**|800<br>800|Negative<br>Positive|
|**Test**|200<br>200|Negative<br>Positive|



Samples of the dataset are depicted in Figure 3.
The dataset is arbitrarily partitioned among each client ( _𝑐_ ∈
{ 2 _,_ 3 _,_ 5 _,_ 7 } )., and then the prediction performance results in the
encrypted-domain are compared with the results of the plain-domain.


**5.2** **Experimental Results**


Table 2 shows the best performance of the conventional CNN
method of COVID-19 Xray scans dataset.
Table 3 shows the prediction performance of the CNN based
classification model with and without encryption. As shown in the
table, when the number of clients varies from 2 to 7, then the overall



**Table 2: Initial results in plain domain without using feder-**
**ated learning**

|Metric|Value|
|---|---|
|Precision<br>Recall<br>F1 Score<br>Accuracy|0.868924<br>0.840000<br>0.836801<br>0.840000|



prediction performance stays relatively stable at about 0.84 in the
proposed training method.
Figure 4 shows the execution times in seconds with three different configuration (i.e. plain, s=128, s=192). As expected, the execution in the encrypted domain is much higher than the plain
domain.

|Parameters No encryption HE, s=128 HE, s=192|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|Col14|Col15|Col16|Col17|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||||||||||||||||||
||||||||||||||||||
||||||||||||||||||
||||||||||||||||||
||||||||||||||||||
||||||||||||||||||
||||||||||||||||||
||||||||||||||||||
||||||||||||||||||
||||||||||||||||||
||||||||||||||||||
||||||||||||||||||
||||||||||||||||||
||||||||||||||||||
||||||||||||||||||
||||||||||||||||||



**Figure 4: Execution time in seconds with the different secu-**
**rity levels.**


**6** **DISCUSSION**


The experimental results in figure 4 provides new insights into the
relationships between different number of clients and execution
time. There is a significant difference in execution time between
plain ( Unencrypted) and encrypted data processes. This exponential differences are due to the complexity of the homomorphic
encryption and processing encrypted data. However the execution
times of different ciphertext modulus values (128,192) are indistinguishable for two clients but, execution time variation is rising
with the growing the number of clients. That being so, there is an
anticipated trade off between execution time and security level of
the models.

For the prediction phase, the test performances of the both encrypted and unencrypted processes are very similar as indicated in
table 3. In fact, similar performances are achieved by each model
with increasing the number of clients. Moreover, for some cases,
results with plain data performs slightly better than the applied


EICC ’22, June 15–16, 2022, Barcelona, Spain Wibawa and Catak, et al.


**Table 3: Prediction performance of the somewhat HE and plain numbers based federated learning models.**


|Clients|Accuracy|F1|Precision|Recall|
|---|---|---|---|---|
|**Clients**|**128**<br>**192**<br>**Plain**|**128**<br>**192**<br>**Plain**|**128**<br>**192**<br>**Plain**|**128**<br>**192**<br>**Plain**|
|2<br>3<br>5<br>7|0.8375<br>0.8400<br>0.8450<br>0.8400<br>0.8400<br>0.8375<br>0.8300<br>0.8325<br>0.8350<br>0.8525<br>0.8450<br>0.8275|0.834132<br>0.837030<br>0.842123<br>0.838040<br>0.836801<br>0.834369<br>0.827078<br>0.829732<br>0.832164<br>0.850776<br>0.842540<br>0.824649|0.867337<br>0.866735<br>0.872128<br>0.857293<br>0.868924<br>0.865112<br>0.853925<br>0.855624<br>0.859288<br>0.869584<br>0.868000<br>0.850277|0.8375<br>0.8400<br>0.8450<br>0.8400<br>0.8400<br>0.8375<br>0.8300<br>0.8325<br>0.8350<br>0.8525<br>0.8450<br>0.8275|



encryption results. For instance, the accuracy results of five clients
indicates that plain versions accomplished better for each metric
namely, accuracy, F1, precision, and Recall.


**7** **CONCLUSION**


Privacy preserving become an essential practice of healthcare institutions as it is mandated by both EU and the US. Federated learning
and homomorphic encryption will play critical role to maintain data
security and model training. With benefitting from both techniques,
the proposed model achieves compatitive performance while there
is a significant trade off for the execution time and number of
clients. The classification metrics, i.e. accuracy, F1. precision and
recall, reaches over %80 using both encrypted and plain data for
each federated learning case.
The privacy attacks will cause immense damages to the security and privacy of the patient information. This will hinder the
advancement in healthcare using data-driven models. Therefore it
is indispensable to take imperative steps to strengthen not only the
safety of the information but also the way data is processed. This
study demonstrated that federated learning with homomorphic
encryption could be successfully applied to enhance data-driven
models by eliminating and minimizing the share of the sensitive
data. It is envisioned that this study could be useful for the scientists and researchers working on the sensitive healthcare data in
multi-party computation settings.


**REFERENCES**


[1] Karim Abouelmehdi, Abderrahim Beni-Hssane, Hayat Khaloufi, and Mostafa
Saadi. 2017. Big data security and privacy in healthcare: A Review. _Procedia_
_Computer Science_ 113 (2017), 73–80.

[2] Abbas Acar, Hidayet Aksu, A Selcuk Uluagac, and Mauro Conti. 2018. A survey on
homomorphic encryption schemes: Theory and implementation. _ACM Computing_
_Surveys (Csur)_ 51, 4 (2018), 1–35.

[3] Aitizaz Ali, Muhammad Fermi Pasha, Jehad Ali, Ong Huey Fang, Mehedi Masud,
Anca Delia Jurcut, and Mohammed A Alzain. 2022. Deep Learning Based Homomorphic Secure Search-Able Encryption for Keyword Search in Blockchain
Healthcare System: A Novel Approach to Cryptography. _Sensors_ 22, 2 (2022),
528.

[4] Mohamed Alloghani, Mohammed M Alani, Dhiya Al-Jumeily, Thar Baker, Jamila
Mustafina, Abir Hussain, and Ahmed J Aljaaf. 2019. A systematic review on
the status and progress of homomorphic encryption technologies. _Journal of_
_Information Security and Applications_ 48 (2019), 102362.

[5] Rodolfo Stoffel Antunes, Cristiano André da Costa, Arne Küderle, Imrana Abdullahi Yari, and Björn Eskofier. 2022. Federated Learning for Healthcare: Systematic
Review and Architecture Proposal. _ACM Transactions on Intelligent Systems and_
_Technology (TIST)_ (2022).

[6] Luca Barbieri, Stefano Savazzi, Mattia Brambilla, and Monica Nicoli. 2022. Decentralized federated learning for extended sensing in 6G connected vehicles.
_Vehicular Communications_ 33 (2022), 100396.

[7] Razvan Bocu and Cosmin Costache. 2018. A homomorphic encryption-based
system for securely managing personal health metrics data. _IBM Journal of_
_Research and Development_ 62, 1 (2018), 1–1.

[8] Zvika Brakerski. 2012. Fully Homomorphic Encryption without Modulus Switching from Classical GapSVP. In _Advances in Cryptology – CRYPTO 2012_, Reihaneh



Safavi-Naini and Ran Canetti (Eds.). Springer Berlin Heidelberg, Berlin, Heidelberg, 868–886.

[9] Ferhat Ozgur Catak, Ismail Aydin, Ogerta Elezaj, and Sule Yildirim-Yayilgan.
2020. Practical Implementation of Privacy Preserving Clustering Methods Using
a Partially Homomorphic Encryption Algorithm. _Electronics_ [9, 2 (2020). https:](https://doi.org/10.3390/electronics9020229)
[//doi.org/10.3390/electronics9020229](https://doi.org/10.3390/electronics9020229)

[10] Kevser Şahinbaş and Ferhat Ozgur Catak. 2021. Secure Multi-Party Computation based Privacy Preserving Data Analysis in Healthcare IoT Systems.
_arXiv e-prints_, Article arXiv:2109.14334 (Sept. 2021), arXiv:2109.14334 pages.
[arXiv:2109.14334 [cs.CR]](https://arxiv.org/abs/2109.14334)

[11] Junfeng Fan and Frederik Vercauteren. 2012. Somewhat Practical Fully Homomorphic Encryption. _IACR Cryptol. ePrint Arch._ 2012 (2012), 144.

[12] Alberto Ibarrondo and Alexander Viand. 2021. Pyfhel: Python for homomorphic
encryption libraries. In _WAHC 2021, 9th Workshop on Encrypted Computing &amp;_
_Applied Homomorphic Cryptography, Associated with the ACM CCS 2021 conference,_
_15 November 2021, Seoul, South Korea_, ACM (Ed.). Seoul.

[13] Peter Kairouz, H Brendan McMahan, Brendan Avent, Aurélien Bellet, Mehdi Bennis, Arjun Nitin Bhagoji, Kallista Bonawitz, Zachary Charles, Graham Cormode,
Rachel Cummings, et al . 2021. Advances and open problems in federated learning.
_Foundations and Trends® in Machine Learning_ 14, 1–2 (2021), 1–210.

[14] Mostefa Kara, Abdelkader Laouid, Mohammed Amine Yagoub, Reinhardt Euler,
Saci Medileh, Mohammad Hammoudeh, Amna Eleyan, and Ahcène Bounceur.
2021. A fully homomorphic encryption based on magic number fragmentation
and El-Gamal encryption: Smart healthcare use case. _Expert Systems_ (2021),
e12767.

[15] A Vijaya Kumar, Mogalapalli Sai Sujith, Kosuri Tarun Sai, Galla Rajesh, and
Devulapalli Jagannadha Sriram Yashwanth. 2020. Secure Multiparty computation
enabled E-Healthcare system with Homomorphic encryption. In _IOP Conference_
_Series: Materials Science and Engineering_, Vol. 981. IOP Publishing, 022079.

[16] Wenqi Li, Fausto Milletarì, Daguang Xu, Nicola Rieke, Jonny Hancox, Wentao
Zhu, Maximilian Baust, Yan Cheng, Sébastien Ourselin, M Jorge Cardoso, et al .
2019. Privacy-preserving federated brain tumour segmentation. In _International_
_workshop on machine learning in medical imaging_ . Springer, 133–141.

[17] Nicola Rieke, Jonny Hancox, Wenqi Li, Fausto Milletari, Holger R Roth, Shadi
Albarqouni, Spyridon Bakas, Mathieu N Galtier, Bennett A Landman, Klaus MaierHein, et al . 2020. The future of digital health with federated learning. _NPJ digital_
_medicine_ 3, 1 (2020), 1–7.

[18] Abhijit Guha Roy, Shayan Siddiqui, Sebastian Pölsterl, Nassir Navab, and Christian Wachinger. 2019. Braintorrent: A peer-to-peer environment for decentralized
federated learning. _arXiv preprint arXiv:1905.06731_ (2019).

[19] Micah J Sheller, G Anthony Reina, Brandon Edwards, Jason Martin, and Spyridon
Bakas. 2018. Multi-institutional deep learning modeling without sharing patient
data: A feasibility study on brain tumor segmentation. In _International MICCAI_
_Brainlesion Workshop_ . Springer, 92–104.

[20] Mir Sajjad Hussain Talpur, Md Zakirul Alam Bhuiyan, and Guojun Wang. 2015.
Shared–node IoT network architecture with ubiquitous homomorphic encryption
for healthcare monitoring. _International Journal of Embedded Systems_ 7, 1 (2015),
43–54.

[21] Haowen Tan, Pankoo Kim, and Ilyong Chung. 2020. Practical homomorphic
authentication in cloud-assisted vanets with blockchain-based healthcare monitoring for pandemic control. _Electronics_ 9, 10 (2020), 1683.

[22] Evert-Ben van Veen. 2018. Observational health research in Europe: understanding the General Data Protection Regulation and underlying debate. _European_
_Journal of Cancer_ [104 (2018), 70–80. https://doi.org/10.1016/j.ejca.2018.09.032](https://doi.org/10.1016/j.ejca.2018.09.032)

[23] Xiaoni Wang and Zhenjiang Zhang. 2015. Data division scheme based on homomorphic encryption in WSNs for health care. _Journal of medical systems_ 39, 12
(2015), 1–7.

[24] Jie Xu, Benjamin S Glicksberg, Chang Su, Peter Walker, Jiang Bian, and Fei
Wang. 2021. Federated learning for healthcare informatics. _Journal of Healthcare_
_Informatics Research_ 5, 1 (2021), 1–19.

[25] Ferhat Özgür Çatak and Ahmet Fatih Mustacoglu. 2018. CPP-ELM: Cryptographically Privacy-Preserving Extreme Learning Machine for Cloud Systems.
_International Journal of Computational Intelligence Systems_ 11 (2018), 33–44. Issue
[1. https://doi.org/10.2991/ijcis.11.1.3](https://doi.org/10.2991/ijcis.11.1.3)


