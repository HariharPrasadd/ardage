## **E FFeL: Ensuring Integrity For Federated Learning**



Somesh Jha Laurens van der


Maaten
UW Madison



Amrita Roy
Chowdhury [∗]


UW Madison


**Abstract**



Chuan Guo



Somesh Jha [†]

Meta AI UW Madison



Meta AI



Federated learning (FL) enables clients to collaborate with a server
to train a machine learning model. To ensure privacy, the server
performs secure aggregation of updates from the clients. Unfortunately, this prevents verification of the well-formedness (integrity)
of the updates as the updates are masked. Consequently, malformed
updates designed to poison the model can be injected without detection. In this paper, we formalize the problem of ensuring _both_
update privacy and integrity in FL and present a new system, EIFFeL, that enables secure aggregation of _verified_ updates. EIFFeL is a
general framework that can enforce _arbitrary_ integrity checks and
remove malformed updates from the aggregate, without violating
privacy. Our empirical evaluation demonstrates the practicality of
EIFFeL. For instance, with 100 clients and 10% poisoning, EIFFeL can
train an MNIST classification model to the same accuracy as that
of a non-poisoned federated learner in just 2 _._ 4s per iteration.


**CCS Concepts**


- **Security and privacy** → **Cryptography** ; **Privacy-preserving**
**protocols** .


**Keywords**


Poisoning Attacks, Input Integrity, Secure Aggregation


**ACM Reference Format:**


Amrita Roy Chowdhury, Chuan Guo, Somesh Jha, and Laurens van der
Maaten. 2022. E FFeL: Ensuring Integrity For Federated Learning. In _Pro-_
_ceedings of the 2022 ACM SIGSAC Conference on Computer and Communica-_
_tions Security (CCS ’22), November 7–11, 2022, Los Angeles, CA, USA._ ACM,
[New York, NY, USA, 21 pages. https://doi.org/10.1145/3548606.3560611](https://doi.org/10.1145/3548606.3560611)


**1** **Introduction**


Federated learning (FL; [ 61 ]) is a learning paradigm for decentralized data in which multiple clients collaborate with a server to train
a machine-learning (ML) model. Each client computes an update
on its _local_ training data and shares it with the server; the server
aggregates the local updates into a _global_ model update. This allows


∗ Work done during internship at Meta AI

- Employed part-time at Meta during this work


Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than ACM
must be honored. Abstracting with credit is permitted. To copy otherwise, or republish,
to post on servers or to redistribute to lists, requires prior specific permission and/or a
fee. Request permissions from permissions@acm.org.
_CCS ’22, November 7–11, 2022, Los Angeles, CA, USA_
© 2022 Association for Computing Machinery.
ACM ISBN 978-1-4503-9450-5/22/11...$15.00
[https://doi.org/10.1145/3548606.3560611](https://doi.org/10.1145/3548606.3560611)



**Security Goal** **Cryptographic Primitive**


Input Privacy Shamir’s Threshold Secret Sharing Scheme [75]


Secret-Shared Non-Interactive Proof [28]
Input Integrity
Verifiable Secret Shares [35]


**Figure 1: EIFFeL performs secure aggregation of** _**verified**_ **inputs in**
**FL. The table lists its security goals and the cryptographic primi-**
**tives we adopt to achieve them.**


the clients to contribute to model training without sharing their
private data. However, the local updates can still reveal information
about a client’s private data [ 11, 63, 65, 95, 97 ]. FL addresses this
by using _secure aggregation_ : clients mask the updates they share,
and the server can recover _only_ the final aggregate in the clear.


A major challenge in FL is that it is vulnerable to Byzantine attacks.
In particular, malicious clients can inject poisoned updates into
the learner with the goal of reducing the global model accuracy

[ 10, 12, 34, 45, 62 ] or implanting backdoors in the model that can
be exploited later [ 5, 26, 90 ]. Even a single malformed update can
significantly alter the trained model [ 15 ]. Thus, ensuring the wellformedness of the updates, _i.e._, upholding their _integrity_, is essential
for ensuring robustness in FL. This problem is especially challenging
in the context of secure aggregation as the individual updates are
masked from the server, which prevents audits on them.


These challenges in FL lead to the research question: _How can a_
_federated learner efficiently verify the integrity of clients’ updates_
_without violating their privacy?_


We formalize this problem by proposing _secure aggregation of veri-_
_fied inputs_ (SAVI) protocols that: ( 1 ) securely verify the integrity
of each local update, ( 2 ) aggregate _only_ well-formed updates, and
( 3 ) release only the final aggregate in the clear. A SAVI protocol allows for checking the well-formedness of updates _without observing_
_them_, thereby ensuring _both_ the privacy and integrity of updates.


We demonstrate the feasibility of SAVI by proposing EIFFeL: a system that instantiates a SAVI protocol that can perform _any integrity_
_check that can be expressed as an arithmetic circuit with public param-_
_eters_ . This provides EIFFeL the flexibility to implement a plethora of
modern ML approaches that ensure robustness to Byzantine attacks
by checking the integrity of per-client updates before aggregating
them [ 5, 31, 54, 76, 83, 84, 92, 93 ]. EIFFeL is a general framework





𝟏 _M_


_u_ 2 = 𝒖



𝒰 = X



.






CCS ’22, November 7–11, 2022, Los Angeles, CA, USA Roy Chowdhury et al.



that empowers a federated learner to deploy (multiple) _arbitrary_
integrity checks of their choosing on the “masked” updates.


EIFFeL uses secret-shared non-interactive proofs (SNIP; [ 28 ]) which
are a type of zero-knowledge proofs that are optimized for the clientserver setting. SNIP, however, requires multiple honest verifiers to
check the proof. EIFFeL extends SNIP to a _malicious_ threat model
by carefully _co-designing its architectural and cryptographic compo-_
_nents_ . Moreover, we develop a suite of optimizations that improve
EIFFeL’s performance by at least 2 _._ 3 × . Our empirical evaluation
of EIFFeL demonstrates its practicality for real-world usage. For
instance, with 100 clients and a poisoning rate of 10%, EIFFeL can
train an MNIST classification model to the same accuracy as that
of a non-poisoned federated learner in just 2 _._ 4 _𝑠_ per iteration.


**2** **Problem Overview**


In this section, we introduce the problem setting, followed by its
threat analysis and an overview of our solution.


**2.1** **Problem Setting**


In FL, multiple parties with distributed data jointly train a _global_
_model_, M, without explicitly disclosing their data to each other. FL
has two types of actors:

- **Clients.** There are _𝑛_ clients where each client, C _𝑖_ _,𝑖_ ∈[ _𝑛_ ], owns
a private dataset, _𝐷_ _𝑖_ . The raw data is never shared, instead, every client computes a local update for M, such as the average
gradient, over the private dataset _𝐷_ _𝑖_ .

- **Server.** There is a single _untrusted_ server, S, who coordinates
the updates from different clients to train M.
A single training iteration in FL consists of the following steps:

- **Broadcast.** The server broadcasts the current parameters of the
model M to all the clients.

- **Local computation.** Each client C _𝑖_ locally computes an update,
_𝑢_ _𝑖_, on its dataset _𝐷_ _𝑖_ .

- **Aggregation.** The server S collects the client updates and aggregates them, U = [�] _𝑖_ ∈[ _𝑛_ ] _[𝑢]_ _𝑖_ [.]

- **Global model update.** The server S updates the global model
M based on the aggregated update U.
In settings where there is a large number of clients, it is typical to
subsample a small subset of clients to participate in a given iteration.
We assume _𝑛_ to denote the number of clients that participate in
each iteration and C denotes the subset of these _𝑛_ clients, which
the server announces at the beginning of the iteration.

**2.2** **Security Goals**


- **Input Privacy (Client’s Goal).** The first goal is to ensure privacy for all _honest_ clients. That is, no party should be able learn
anything about the raw input (update) _𝑢_ _𝑖_ of an honest client _𝐶_ _𝑖_,
other than what can be learned from the final aggregate U .

- **Input Integrity (Server’s Goal).** The server S is motivated to
ensure that the individual updates from each client are wellformed. Specifically, the server has a _public_ validation predicate,
Valid(·), that defines a syntax for the inputs (updates). An input
(update) _𝑢_ is considered valid and, hence, passes the integrity
check iff Valid( _𝑢_ ) = 1 . For instance, any per-client update check,
such as Zeno++ [ 93 ], can be a good candidate for Valid(·) (we
evaluate four state-of-the-art validation predicates in Sec. 7.2).



We assume that the honest clients, denoted by C _𝐻_ : ( 1 ) follow the
protocol correctly, _and_ ( 2 ) have well-formed inputs. We require the
second condition because, in case the input of an honest client does
not pass the integrity check (which can be verified locally since
Valid(·) is public), the client has no incentive to participate in the
training iteration.


**2.3** **Threat Model**


We consider a _malicious adversary_ threat model:

- **Malicious Server.** We consider a malicious server that can de
viate from the protocol arbitrarily with the aim of recovering the
raw updates _𝑢_ _𝑖_ for _𝑖_ ∈[ _𝑛_ ] (see Remark 1 later for more details).

- **Malicious Clients.** We also consider a set of _𝑚_ malicious clients,
C _𝑀_ . Malicious clients can arbitrarily deviate from the protocol
with the goals of: (1) sending malformed inputs to the server and
thus, compromising the final aggregate; (2) failing the integrity
check of an honest client that submits well-formed updates; (3)
violating the privacy of an honest client, potentially in collusion
with the server.


**2.4** **Solution Overview**


Prior work has mostly focused on ensuring input privacy via secure
aggregation, _i.e._, securely computing the aggregate U = [�] C _𝑖_ ∈C _[𝑢]_ _𝑖_ [.]
Motivated by the above problem setting and threat analysis, we
introduce a new type of FL protocol, called _secure aggregation with_
_verified inputs_ (SAVI), that ensures _both_ input privacy and integrity.
The goal of a SAVI protocol is to securely aggregate _only_ wellinformed inputs.
In order to demonstrate the feasibility of SAVI, we propose EIFFeL:
a system that instantiates a SAVI protocol for any Valid(·) that can
be expressed as an arithmetic circuit with public parameters (Fig.
1). EIFFeL ensures input privacy by using Shamir’s threshold secret
sharing scheme [ 75 ] (Sec. 4.1). Input integrity is guaranteed via SNIP
and verifiable secret shares (VSS) which validates the correctness
of the secret shares (Sec. 4.1). The key ideas are:

- SNIP requires multiple honest verifiers. EIFFeL enables this in
a single-server setting by having the clients act as the verifiers
for each other under the supervision of the server (in Fig. 2b,
verifiers are marked by ).

- EIFFeL extends SNIP to the malicious threat model to account

for the malicious clients (verifiers). Our key observation is that
using a threshold secret sharing scheme creates multiple subsets
of clients that can emulate the SNIP verification protocol. The
server uses this redundancy to robustly verify the proofs and
aggregate updates with verified proofs _only_ (Fig. 2c and 2d).


**3** **Secure Aggregation with Verified Inputs**


Below, we provide the formal definition of a _secure aggregation with_
_verified inputs_ (SAVI) protocol.


Definition 1. _Given a public validation predicate_ _Valid_ (·) _and se-_
_curity parameter_ _𝜅_ _, a protocol_ Π( _𝑢_ 1 _,_ - · · _,𝑢_ _𝑛_ ) _is a secure aggregation_
_with verified inputs (SAVI) protocol if:_


- _**Integrity.**_ _The output of the protocol,_ _out_ _, returns the aggregate_
_of a subset of clients,_ C _Valid_ _, such that all clients in_ C _Valid_ _have_
_well-formed inputs._


E FFeL: Ensuring Integrity For Federated Learning CCS ’22, November 7–11, 2022, Los Angeles, CA, USA



**(a) EIFFeL consists of multiple clients**
C **and a server** S **with a public vali-**
**dation predicate Valid** (·) **that defines**
**the integrity check. A client** C _𝑖_ **needs**
**to provide a proof** _𝜋_ _𝑖_ **for Valid** ( _𝑢_ _𝑖_ ) = 1
**(Round 1).**



**(b) For checking the proof** _𝜋_ _𝑖_ **, all**
**other clients** C \ _𝑖_ **act as the verifiers**
**under the supervision of** S **.** _𝐶_ _𝑖_ **splits**
**its update** _𝑢_ _𝑖_ **and proof** _𝜋_ _𝑖_ **using**
**Shamir’s scheme with threshold** _𝑚_ +1
**and shares it with** C \ _𝑖_ **(Round 2).**



**(c) Conceptually, any set of** _𝑚_ + 1
**clients in** C \ _𝑖_ **can emulate the SNIP**
**verification protocol. The server uses**
**this redundancy to** _**robustly**_ **verify**
**the proof (Round 3).**



**(d) The clients only aggregate the**
**shares of well-formed updates and**
**the resulting aggregate is revealed to**
**the server (Round 4).**



**Figure 2: High-level overview of EIFFeL. See Sec. 2.4 for key ideas, and Sec. 4.4 for a detailed description of the system.**



Pr� _out_ = U _Valid_ � ≥ 1 − negl( _𝜅_ ) _where_ U _Valid_ = ∑︁ _𝑢_ _𝑖_

C _𝑖_ ∈C _Valid_


_for all_ C _𝑖_ ∈C _Valid_ _we have Valid_ ( _𝑢_ _𝑖_ ) = 1


C _𝐻_ ⊆C _Valid_ ⊆C _._ (1)


- _**Privacy.**_ _For a set of malicious clients_ C _𝑀_ _and a malicious server_

S _, there exists a probabilistic polynomial-time (P.P.T.) simulator_
Sim(·) _such that:_


Real Π �{ _𝑢_ C _𝐻_ } _,_ Ω C _𝑀_ ∪S � ≡ _𝐶_ Sim�U _𝐻_ _,_ C _𝐻_ _,_ Ω C _𝑀_ ∪S �

_where_ U _𝐻_ = ∑︁ _𝑢_ _𝑖_ _._ (2)

C _𝑖_ ∈C _𝐻_


{ _𝑢_ C _𝐻_ } _denotes the input of all the honest clients,_ Real Π _denotes_
_a random variable representing the joint view of all the parties_
_in_ Π _’s execution,_ Ω C _𝑀_ ∪S _indicates a polynomial-time algorithm_
_implementing the “next-message” function of the parties in_ C _𝑀_ ∪S
_(see App. 11.5), and_ ≡ _𝐶_ _denotes computational indistinguishability._


From Def. 1, the output of a SAVI protocol is of the form:







U _𝑣𝑎𝑙𝑖𝑑_ = U _𝐻_

����


well-formed updates of
_all_ honest clients C _𝐻_



+ _𝑢_ _𝑖_ _._
∑︁

C _𝑖_ ∈C Valid \C _𝐻_


� **������������** �� **������������** �


well-formed updates of
some malicious clients



(3)



The clients in C Valid \ C _𝐻_ are clients who have submitted wellformed inputs but can behave maliciously otherwise ( _e.g._, by violating input privacy/integrity of honest clients).


The privacy constraint of the SAVI protocol means that a simulator
Sim can generate the views of all parties with just access to the
list of the honest clients C _𝐻_ and their aggregate U _𝐻_ . Note that
Sim takes U _𝐻_ as an input instead of the protocol output U Valid .
This is because the clients in C Valid \ C _𝐻_, by virtue of being malicious, can behave arbitrarily and announce their updates to reveal
U _𝐻_ = U Valid − [�] C _𝑖_ ∈C Valid \C _𝐻_ _[𝑢]_ _𝑖_ [. Thus,][ SAVI][ ensures that nothing can]
be learned about the input _𝑢_ _𝑖_ of an honest client C _𝑖_ ∈C _𝐻_ except:


- that _𝑢_ _𝑖_ is well-formed, _i.e._, Valid( _𝑢_ _𝑖_ ) = 1,

- anything that can be learned from the aggregate U _𝐻_ .



**4** **EIFFeL System Description**


This section introduces EIFFeL: the system we propose to perform
secure aggregation of verified inputs.


**4.1** **Cryptographic Building Blocks**


**Arithmetic Circuit.** An arithmetic circuit, C : F _[𝑘]_ ↦→ F, represents
a computation over a finite field F . Conceptually, it is similar to a
Boolean circuit but it uses finite field addition, multiplication and
multiplication-by-constant instead of OR, AND, and NOT gates.


CCS ’22, November 7–11, 2022, Los Angeles, CA, USA Roy Chowdhury et al.



**Shamir’s** _𝑡_ **-out-of-** _𝑛_ **Secret Sharing Scheme [75]** allows distributing a secret _𝑠_ among _𝑛_ parties such that: (1) the complete secret can
be reconstructed from any combination of _𝑡_ shares; (2) any set of
_𝑡_ − 1 or fewer shares reveals no information about _𝑠_ where _𝑡_ is the

_threshold_ of the secret sharing scheme. The scheme is parameterized
over a finite field F and consists of two algorithms:


$

- {( _𝑖,𝑠_ _𝑖_ )} _𝑖_ ∈ _𝑃_ ←− SS.share( _𝑠, 𝑃,𝑡_ ) . Given a secret _𝑠_ ∈ F, a set of _𝑛_
unique field elements _𝑃_ ∈ F _[𝑛]_ and a threshold _𝑡_ with _𝑡_ ≤ _𝑛_, this
algorithm constructs _𝑛_ shares. The algorithm chooses a random
polynomial _𝑝_ ∈ F[ _𝑋_ ] such that _𝑝_ (0) = _𝑠_ and generates the shares
as ( _𝑖, 𝑝_ ( _𝑖_ )) _,𝑖_ ∈ _𝑃_ .

- _𝑠_ ← SS.recon({( _𝑖,𝑠_ _𝑖_ ) _𝑖_ ∈ _𝑄_ }) . Given the shares corresponding to a
subset _𝑄_ ⊆ _𝑃,_ | _𝑄_ | ≥ _𝑡_, the reconstruction algorithm recovers the

secret _𝑠_ .


Shamir’s secret sharing scheme is _linear_, which means a party
can _locally_ perform: ( 1 ) addition of two shares, ( 2 ) addition of a
constant, and (3) multiplication by a constant.
Shamir’s secret sharing scheme is closely related to Reed-Solomon
error correcting codes [ 55 ], which is a group of polynomial-based
error correcting codes. The share generation is similar to (nonsystemic) message encoding in these codes which can successfully
recover a message even in the presence of errors and erasures
(message dropouts). Consequently, we can leverage Reed-Solomon
decoding for robust reconstruction of Shamir’s secret shares:

- _𝑠_ ← SS.robustRecon({( _𝑖,𝑠_ _𝑖_ )} _𝑖_ ∈ _𝑄_ ) . Shamir’s secret sharing scheme
results in a [ _𝑛,𝑡,𝑛_ − _𝑡_ + 1] Reed-Solomon code that can tolerate up to _𝑞_ errors and _𝑒_ erasures (message dropouts) such that
2 _𝑞_ + _𝑒_ _< 𝑛_ − _𝑡_ + 1 . Given any subset of _𝑛_ − _𝑒_ shares _𝑄_ ⊆ _𝑃,_ | _𝑄_ | ≥ _𝑛_ − _𝑒_
with up to _𝑞_ errors, any standard Reed Solomon decoding algorithm [ 13 ] can robustly reconstruct _𝑠_ . EIFFeL uses Gao’s decoding
algorithm [37].


_Verifiable secret sharing scheme_ is a related concept where the
scheme has an additional property of _verifiability_ . Given a share
of the secret, a party must be able to check whether it is indeed
a valid share. If a share is valid, then there exists a unique secret
which will be the output of the reconstruction algorithm when run
on any _𝑡_ distinct valid shares. Formally:


- 1/0 ← SS.verify(( _𝑖, 𝑣_ ) _,_ Ψ)) . The verification algorithm inputs a share
and a check string Ψ _𝑠_ such that


∀ _𝑉_ ⊂ F × F where | _𝑉_ | = _𝑡,_ ∃ _𝑠_ ∈ F s.t.


(∀( _𝑖, 𝑣_ ) ∈ _𝑉,_ SS.verify (( _𝑖, 𝑣_ ) _,_ Ψ _𝑠_ ) = 1) =⇒ SS.recon ( _𝑉_ ) = _𝑠_


The share construction algorithm is augmented to output the
check string as ({( _𝑖,𝑠_ _𝑖_ ) _𝑖_ ∈ _𝑃_ } _,_ Ψ _𝑠_ ) ← SS.share( _𝑠, 𝑃,𝑡_ ) .

For EIFFeL, we use the non-interactive verification scheme by Feldman [35] (details in App. 11.1).


**Key Agreement Protocol.** A key agreement protocol consists of
a tuple of the following three algorithms:


- ( _𝑝𝑝_ ) ←− $ KA.param(1 _𝜅_ ) . The parameter generation algorithm samples a set of public parameters _𝑝𝑝_ with security parameter _𝜅_ .

$

- ( _𝑝𝑘,𝑠𝑘_ ) ←− KA.gen( _𝑝𝑝_ ) . The key generation algorithm samples a
public/secret key pair from the public parameters.



Sim _𝜋_ ( Valid (·) _,_ {([ _𝑥_ ] _𝑖_ _,_ [ _𝜋_ ] _𝑖_ )} _𝑖_ ∈V¯ [) ≡] [View] _𝜋,_ V [¯] [(] [Valid] [(·)] _[,𝑥]_ [)] _[.]_


Thus, SNIP allows the verifiers to collaboratively check – without ever accessing the prover’s private data in the clear – that the




- _𝑠𝑘_ _𝑖𝑗_ ← KA.agree( _𝑝𝑘_ _𝑖_ _,𝑠𝑘_ _𝑗_ ) . The key agreement protocol receives a
public key _𝑝𝑘_ _𝑖_ and a secret key _𝑠𝑘_ _𝑗_ as input and generates the
shared key _𝑠𝑘_ _𝑖𝑗_ .


**Authenticated Encryption** provides confidentiality and integrity
guarantees for messages exchanged between two parties. It consists
of a tuple of three algorithms as follows:

- _𝑘_ ←− $ AE.gen(1 _𝜅_ ) . The key generation algorithm that outputs a
private key _𝑘_ where _𝜅_ is the security parameter.

$

- ~~_𝑥_~~ ←− AE.enc( _𝑘,𝑥_ ) . The encryption algorithm takes as input a key
_𝑘_ and a message _𝑥_, and outputs a ciphertext ~~_𝑥_~~ .

- _𝑥_ ← AE.dec( _𝑘,_ ~~_𝑥_~~ ~~)~~ . The decryption algorithm takes as input a ciphertext and a key and outputs either the original plaintext, or a
special error symbol ⊥ on failure.

**Secret-shared Non-interactive Proofs.** The secret-shared non
interactive proof (SNIP) [ 28 ] is an information-theoretic zero-knowledge proof for distributed data (Fig. 3). SNIP is designed for a multiverifier setting where the private data is distributed or secret-shared
among the verifiers. Specifically, SNIP relies on an additive secret
sharing scheme over a field F as described below. A secret _𝑠_ ∈ F is
split into _𝑘_ random shares ([ _𝑠_ ] 1 _,_ - · · _,_ [ _𝑠_ ] _𝑘_ ) such that [�] _[𝑘]_ _𝑖_ =1 [[] _[𝑠]_ []] _[𝑖]_ [=] _[ 𝑠]_ [.]
A subset of up to _𝑘_ − 1 shares reveals _no_ information about the
secret _𝑠_ . The additive secret-sharing scheme is linear as well.


_SNIP Setting._ SNIP considers _𝑘_ ≥ 2 verifiers {V _𝑖_ } _,𝑖_ ∈[ _𝑘_ ] and a prover
P with a private vector _𝑥_ ∈ F _[𝑑]_ . All parties also hold a _public_ arithmetic circuit representing a validation predicate Valid : F _[𝑑]_ ↦→ F . Let
M be the number of multiplication gates in Valid(·) . F is chosen such
that 2M ≪|F| . The prover P splits _𝑥_ into _𝑘_ shares {[ _𝑥_ 1 ] _,_ - · · _,_ [ _𝑥_ _𝑘_ ]} .
Next, they generate _𝑘_ proof strings [ _𝜋_ ] _𝑖_ _,𝑖_ ∈[ _𝑘_ ] based on Valid(·)
and shares ([ _𝑥_ _𝑖_ ] _,_ [ _𝜋_ ] _𝑖_ ) with every verifier V _𝑖_ (Fig. 3a).
The prover’s goal is to convince the verifiers that, indeed, Valid( _𝑥_ ) = 1 .
The prover does so via proof strings [ _𝜋_ ] _𝑖_ _,𝑖_ ∈[ _𝑘_ ], that do not reveal
anything else about _𝑥_ . After receiving the proof, the verifiers gossip
with each other to conclude either that Valid( _𝑥_ ) = 1 (the verifiers

“
Accept _𝑥_ ") or not (“ Reject _𝑥_ ”, Figs. 3b and 3c). Formally, SNIP satisfies
the following security properties:


- _Completeness._ If all parties are honest and Valid( _𝑥_ ) = 1, then the
verifiers will accept _𝑥_ .


∀ _𝑥_ ∈ F s.t. Valid ( _𝑥_ ) = 1 : Pr _𝜋_ [ Accept _𝑥_ ] = 1 _._


- _Soundness_ . If all verifiers are honest, and if Valid( _𝑥_ ) = 0, then for all
malicious provers, the verifiers will reject _𝑥_ with overwhelming
probability.


∀ _𝑥_ ∈ F s.t. Valid ( _𝑥_ ) = 0 : Pr _𝜋_ �Reject _𝑥_ � ≥ 1 − (2 _𝑀_ −2) / |F | _._


- _Zero knowledge._ If the prover and at least one verifier are honest,
then the verifiers learn nothing about _𝑥_, except that Valid( _𝑥_ ) = 1 .
Formally, when Valid( _𝑥_ ) = 1, there exists a simulator Sim(·) that
can simulate the view of the protocol execution for every proper
subset of verifiers:



∀ _𝑥_ s.t. Valid ( _𝑥_ ) = 1 and ∀V ⊂ [¯]



_𝑘_
�



V _𝑖_ we have

_𝑖_ =1


E FFeL: Ensuring Integrity For Federated Learning CCS ’22, November 7–11, 2022, Los Angeles, CA, USA



**(a) Prover sends secret shares of its input and**
**the SNIP proof to multiple verifiers.**



**(b) The verifiers gossip among themselves** **(c) The check passes successfully if all veri-**
**and check the proof.** **fiers are honest.**

**Figure 3: High-level overview of a secret-shared non-interactive proof (SNIP; [28]).**



**(b) The verifiers gossip among themselves**
**and check the proof.**



prover’s submission is, indeed, well-formed. SNIP works in two
stages as follows:
( 1 ) _Generation of Proof._ For generating the proof, the prover P first
evaluates the circuit Valid(·) on its input _𝑥_ to obtain the value of every wire in the arithmetic circuit corresponding to the computation
of Valid( _𝑥_ ) . Using these wire values, P constructs three polynomials _𝑓_, _𝑔_, and _ℎ_ of the lowest possible degrees such that _ℎ_ = _𝑓_ - _𝑔_ and
_𝑓_ ( _𝑗_ ) _,𝑔_ ( _𝑗_ ) and _ℎ_ ( _𝑗_ ) _, 𝑗_ ∈ �M� encode the values of the two input wires
and one output wire of the _𝑗_ -th multiplication gate, respectively.
P also samples a single set of Beaver’s multiplication triples [ 7 ]:
( _𝑎,𝑏,𝑐_ ) ∈ F [3] such that _𝑎_ - _𝑏_ = _𝑐_ ∈ F . Finally, it generates the shares of
the proof, [ _𝜋_ ] _𝑖_ = [�] [ _ℎ_ ] _𝑖_ _,_ ([ _𝑎_ ] _𝑖_ _,_ [ _𝑏_ ] _𝑖_ _,_ [ _𝑐_ ] _𝑖_ ) [�], which consists of:

- shares of the coefficients of the polynomial _ℎ_, denoted by [ _ℎ_ ] _𝑖_,

- shares of the Beaver’s triples, ([ _𝑎_ ] _𝑖_ _,_ [ _𝑏_ ] _𝑖_ _,_ [ _𝑐_ ] _𝑖_ ) ∈ F [3] .


The prover then sends the respective shares of the input and the
proof ([ _𝑥_ ] _𝑖_ _,_ [ _𝜋_ ] _𝑖_ ) to each of the verifiers V _𝑖_ .


( 2 ) _Verification of Proof_ . To verify that Valid( _𝑥_ ) = 1 and hence, accept
the input _𝑥_, the verifiers need to check two things:

- check that the value of final output wire of the computation,

Valid( _𝑥_ ), denoted by _𝑤_ _[𝑜𝑢𝑡]_ is indeed 1, and

- check the consistency of P’s computation of Valid( _𝑥_ ) .
To this end, each verifier V _𝑖_ _locally_ constructs the shares of every
wire in Valid( _𝑥_ ) via affine operations on the shares of the private input [ _𝑥_ ] _𝑖_ and [ _ℎ_ ] _𝑖_ . Next, V _𝑖_ broadcasts a summary [ _𝜎_ ] _𝑖_ = ([ _𝑤_ _[𝑜𝑢𝑡]_ ] _𝑖_ _,_ [ _𝜆_ ] _𝑖_ ),
where [ _𝑤_ _[𝑜𝑢𝑡]_ ] _𝑖_ is V _𝑖_ ’s share of the output wire of the circuit and

[ _𝜆_ ] _𝑖_ is a share of a random digest that the verifier computes from
the shares of the other wire values and the proof string [ _𝜋_ ] _𝑖_ . Using
these summaries, the verifiers check the proof as follows:

- For checking the output wire, the verifiers can reconstruct its
exact value from all the broadcasted shares _𝑤_ _[𝑜𝑢𝑡]_ = [�] _[𝑘]_ _𝑖_ =1 [[] _[𝑤]_ _[𝑜𝑢𝑡]_ []] _[𝑖]_
and check whether _𝑤_ _[𝑜𝑢𝑡]_ = 1 . This would imply that Valid( _𝑥_ ) = 1 .

- The circuit consistency check is more involved and is performed
using the random digest _𝜆_ . First, V _𝑖_ _locally_ computes the shares
of the polynomials _𝑓_ and _𝑔_ (denoted as [ _𝑓_ ] _𝑖_ and [ _𝑔_ ] _𝑖_ ). To verify
the consistency of the circuit evaluation, the verifiers need to
check that the shares [ _ℎ_ ] _𝑖_ sent by the prover P are of the correct
polynomial, _i.e._, confirm that _𝑓_  - _𝑔_ = _ℎ_ . For this, SNIP uses the
Schwartz-Zippel polynomial identity test [ 74, 98 ]. Specifically,
verifiers reconstruct _𝜆_ = [�] _[𝑘]_ 1=1 [[] _[𝜆]_ []] _[𝑖]_ [from the broadcasted shares]
and test whether _𝜆_ = _𝑟_ ( _𝑓_ ( _𝑟_ ) · _𝑔_ ( _𝑟_ ) − _ℎ_ ( _𝑟_ )) = 0 on a randomly selected _𝑟_ ∈ F . The computation of the share of the random digest

[ _𝜆_ ] _𝑖_ uses the shares of Beaver’s triples ([ _𝑎_ ] _𝑖_ _,_ [ _𝑏_ ] _𝑖_ _,_ [ _𝑐_ ] _𝑖_ ) .
A more detailed description of the SNIP protocol is in App. 11.1.

**4.2** **System Building Blocks**
**Public Validation Predicate.** EIFFeL requires a public validation
predicate Valid(·), expressed by an arithmetic circuit, that captures



the notion of update well-formedness. In principle, any per-client
update robustness test [ 5, 31, 54, 76, 83, 84, 93 ] from the ML literature can be a suitable candidate. The parameters of the test (for
instance, threshold _𝜌_ for a norm bound check Valid( _𝑢_ ) = I[∥ _𝑢_ ∥ 2 _< 𝜌_ )
can be computed from a clean, public dataset D _𝑃_ that is available to
the server S . This assumption of a clean, public dataset is common
in both ML [ 24, 45, 93 ] as well as privacy literature [ 6, 8, 57 ]. The
dataset can be small and obtained by manual labeling [60].
**Public Bulletin Board.** EIFFeL assumes the availability of a public
bulletin board B that is accessible to all the parties, similar to prior
work [ 17, 45, 72 ]. In practice, the bulletin B can be implemented as
an append-only log hosted at a public web address where every message and its sender is visible. Every party in EIFFeL has read/write
access to it. We use the bulletin B as a tool for broadcasting [ 21, 30 ].


**4.3** **EIFFeL Design Goals**


In terms of the design, EIFFeL should:


- provide _flexibility in the choice of integrity checks._

- be _compatible with the existing FL infrastructure in deployment._

- be _efficient_ in performance.


**4.4** **EIFFeL Workflow**


The goal of EIFFeL is to instantiate a secure aggregation with verified inputs (SAVI) protocol in FL. For a given public validation
predicate Valid(·), EIFFeL checks the integrity of every client update using SNIP and outputs the aggregate of _only_ well-formed
updates, _i.e._, Valid( _𝑢_ ) = 1 . To implement SNIP for our setting, EIFFeL introduces two main ideas:


CCS ’22, November 7–11, 2022, Los Angeles, CA, USA Roy Chowdhury et al.



The full protocol is presented in Fig. 4. The protocol involves a
setup phase followed by four rounds.


**Setup Phase.** In the setup phase, all parties are initialized with the
system-wide parameters, namely the security parameter _𝜅_, the number of clients _𝑛_ out of which _only_ _𝑚_ _<_ ⌊ _[𝑛]_ [−] ~~3~~ [1] [⌋] [can be malicious, public]

$
parameters for the key agreement protocol _𝑝𝑝_ ←− KA.param( _𝜅_ ), and
a field F where |F| ≥ 2 _[𝜅]_ . EIFFeL works in a synchronous protocol between the server S and the _𝑛_ clients in four rounds. To prevent the
server from simulating an arbitrary number of clients, the clients
register themselves with a specific user ID on the public bulletin
board B and are authenticated with the help of standard public
key infrastructure (PKI). The bulletin board B allows parties to
register IDs only for themselves, preventing impersonation. More
concretely, the PKI enables the clients to register identities (public
keys), and sign messages using their identity (associated secret
keys), such that others can verify this signature, but cannot impersonate them [ 46 ]. We omit this detail for the ease of exposition.
For notational simplicity, we assume that each client C _𝑖_ is assigned
a unique logical ID in the form of an integer _𝑖_ in [ _𝑛_ ] . Each client
holds as input a _𝑑_ -dimensional vector _𝑢_ _𝑖_ ∈ F _[𝑑]_ representing its local
update. All clients have a private, authenticated communication
channel with the server S . Additionally, every party (clients and
server) has read and write access to the public bulletin B via authenticated channels. For every client C _𝑖_, the server S maintains
a list, Flag[ _𝑖_ ], of all clients that have flagged C _𝑖_ as malicious. All
Flag[ _𝑖_ ] lists are initialized to be empty lists.


**Round 1 (Announcing Public Information).** In the first round,
all the parties announce their public information relevant to the protocol on the public bulletin B . Specifically, each client C _𝑖_ generates

$
its key pair ( _𝑝𝑘_ _𝑖_ _,𝑠𝑘_ _𝑖_ ) ←− KA.gen( _𝑝𝑝_ ) and advertises the public key
_𝑝𝑘_ _𝑖_ on the public bulletin B . The server S publishes the validation
predicate Valid(·) on B.


**Round 2 (Generate and Distribute Proofs).** Every client generates shares of its private update _𝑢_ _𝑖_ and the proof _𝜋_ _𝑖_, and distributes these shares to the other clients C \ _𝑖_ . First, client C _𝑖_ generates a common pairwise encryption key _𝑠𝑘_ _𝑖𝑗_ for every other client
C _𝑗_ ∈C \ _𝑖_ using the key agreement protocol, _𝑠𝑘_ _𝑖𝑗_ ← KA.agree( _𝑠𝑘_ _𝑖_ _, 𝑝𝑘_ _𝑗_ ) .
Next, the client generates the secret shares of its private update

$
{(1 _,𝑢_ _𝑖_ 1 ) _,_ - · · _,_ ( _𝑛,𝑢_ _𝑖𝑛_ ) _,_ Ψ _𝑢_ _𝑖_ } ←− SS.share( _𝑢,_ [ _𝑛_ ] _,𝑚_ + 1) . The sharing of _𝑢_ _𝑖_
is performed dimension-wise; we abuse notations and denote the
_𝑗_ -th such share by ( _𝑗,𝑢_ _𝑖𝑗_ ) _, 𝑗_ ∈[ _𝑛_ ] . Note that the client C _𝑖_ generates
a share ( _𝑖,𝑢_ _𝑖𝑖_ ) for _itself_ as well which will be used later in the
protocol. Next, the client C _𝑖_ generates the proof for the computation Valid( _𝑢_ _𝑖_ ) = 1 . Specifically, it computes the polynomials _𝑓_ _𝑖_ _,𝑔_ _𝑖_,
and _ℎ_ _𝑖_ = _𝑓_ _𝑖_ - _𝑔_ _𝑖_ and samples a set of Beaver’s multiplication triples
( _𝑎_ _𝑖_ _,𝑏_ _𝑖_ _,𝑐_ _𝑖_ ) ∈ F [3] _,𝑎_ _𝑖_ - _𝑏_ _𝑖_ = _𝑐_ _𝑖_ ∈ F . Since the other clients will verify the
proof, client C _𝑖_ then splits the proof to generate shares _𝜋_ _𝑖𝑗_ = [�] ( _𝑗,ℎ_ _𝑖𝑗_ ) _,_

( _𝑗,𝑎_ _𝑖𝑗_ ) _,_ ( _𝑗,𝑏_ _𝑖𝑗_ ) _,_ ( _𝑗,𝑐_ _𝑖𝑗_ ) [�] for every other client C _𝑗_ ∈C \ _𝑖_ . The shares
themselves are generated via {(1 _,ℎ_ _𝑖_ 1 ) _,_ - · · _,_ ( _𝑖_ − 1 _,ℎ_ _𝑖_ ( _𝑖_ −1) ) _,_ ( _𝑖_ + 1 _,ℎ_ _𝑖_ ( _𝑖_ +1) ) _,_

$

- · · _,_ ( _𝑛,ℎ_ _𝑖𝑛_ ) _,_ Ψ _ℎ_ _𝑖_ } ←− SS.share( _ℎ_ _𝑖_ _,_ [ _𝑛_ ] \ _𝑖,𝑚_ + 1), and so on. Finally, the

client encrypts the proof strings (shares of the update _𝑢_ _𝑖_ and the
proof _𝜋_ _𝑖_ ) using the corresponding pairwise secret key, ( _𝑗,𝑢_ _𝑖𝑗_ )||( _𝑗, 𝜋_ _𝑖𝑗_ )

$
←− AE.enc [�] _𝑠𝑘_ _𝑖𝑗_ _,_ ( _𝑗,𝑢_ _𝑖𝑗_ )||( _𝑗, 𝜋_ _𝑖𝑗_ ) [�], and publishes the encrypted proof
strings on the public bulletin B . The client also publishes the check



strings Ψ _𝑢_ _𝑖_ and Ψ _𝜋_ _𝑖_ = (Ψ _ℎ_ _𝑖_ _,_ Ψ _𝑎_ _𝑖_ _,_ Ψ _𝑏_ _𝑖_ _,_ Ψ _𝑐_ _𝑖_ ) for verifying the validity of
the shares of _𝑢_ _𝑖_ and _𝜋_ _𝑖_, respectively.


**Round 3 (Verify Proof)** . In this round, every client C _𝑖_ partakes
in the verification of the proofs _𝜋_ _𝑗_ of all other clients C _𝑗_ ∈C \ _𝑖_,
under the supervision of the server S . The goal of the server is to
identify the malicious clients, C _𝑀_ . To this end, the server maintains
a (partial) list, C [∗] (initialized as an empty list), of clients it has so
far identified as malicious. The proof-verification round consists of
three phases as follows:


( _𝑖_ ) _Verifying the validity of the secret shares_ . First, every client C _𝑖_
downloads and decrypts their shares from the bulletin B, ( _𝑖,𝑢_ _𝑗𝑖_ )||( _𝑖, 𝜋_ _𝑗𝑖_ )
← AE.dec [�] _𝑠𝑘_ _𝑖𝑗_ _,_ ( _𝑖,𝑢_ _𝑗𝑖_ )||( _𝑖, 𝜋_ _𝑗𝑖_ ) [�] _,_ ∀C _𝑗_ ∈C \ _𝑖_ . Additionally, C _𝑖_ downloads
the check strings (Ψ _𝑢_ _𝑖_ _,_ Ψ _𝜋_ _𝑖_ ) and verifies the validity of the shares. If
the shares from any client C _𝑗_ :

- fail to be decrypted, _i.e._, AE.dec(·) outputs ⊥, OR

- fail to pass the verification, _i.e._, SS.verify(·) returns 0,

C _𝑖_ flags C _𝑗_ on the bulletin B . Every time a client C _𝑖_ flags another
client C _𝑗_, the server updates the corresponding list Flag[ _𝑗_ ]← Flag[ _𝑗_ ] ∪C _𝑖_ .
If |Flag[ _𝑗_ ]| ≥ _𝑚_ + 1, the server S marks C _𝑗_ as malicious: C [∗] ←C [∗] ∪C _𝑗_ .
The server can do so because the pigeon hole principle implies that
C _𝑗_ must have sent an invalid share to at least one honest client;
hence, the correctness of the value recovered from that client’s
shares cannot be guaranteed. In case 1 ≤|Flag[ _𝑗_ ]| ≤ _𝑚_, the server
supervises the following actions. Suppose client C _𝑖_ has flagged
client C _𝑗_ . Client C _𝑗_ then reveals the shares for C _𝑖_, [�] ( _𝑖,𝑢_ _𝑗𝑖_ ) _,_ ( _𝑖, 𝜋_ _𝑗𝑖_ ) [�] in
the clear (on bulletin B ) for the server S (or anyone else) to verify
using SS.Verify(·) . If that verification passes, C _𝑖_ is instructed by the
server to use the released shares for its computations. Otherwise,
C _𝑗_ is marked as malicious by the server S . Note that this does not
lead to privacy violation for an honest client since at most _𝑚_ shares
corresponding to the _𝑚_ malicious clients would be revealed (see Sec.
5). If a client C _𝑖_ flags ≥ _𝑚_ + 1 other clients, S marks C _𝑖_ as malicious.
Thus, at this point every client on the list C [∗] has either

- provided invalid shares to at least one honest client, OR

- flagged an honest client.
In other words, every client who is _not_ in C [∗], C _𝑖_ ∈C \ C [∗], is guaranteed to have submitted at least _𝑛_ − _𝑚_ −1 valid shares for the honest

clients in C _𝐻_ \ C _𝑖_ (see Sec. 5 for details). Additionally, the server
cannot be tricked into marking an honest client as malicious, _i.e._,
EIFFeL ensures C [∗] ∩C _𝐻_ = ∅ (see Sec. 5). The server S publishes C [∗]

on the bulletin B.


( _𝑖𝑖_ ) _Computation of proof summaries by clients._ For this phase, the
server S advertises a random value _𝑟_ ∈ F on the bulletin B . Next, a
client C _𝑖_ proceeds to distill the proof strings of all clients _not_ in C [∗] to
generate summaries for the server S . Specifically, client C _𝑖_ prepares
a proof summary _𝜎_ _𝑗𝑖_ = [�] ( _𝑖,𝑤_ _[𝑜𝑢𝑡]_ _𝑗𝑖_ [)] _[,]_ [ (] _[𝑖, 𝜆]_ _[𝑗𝑖]_ [)][�] [for] [ ∀C] _[𝑗]_ [∈C \ (C] [∗] [∪C] _[𝑖]_ [)] [ as]
per the description in the previous section, and publishes it on B .


( _𝑖𝑖𝑖_ ) _Verification of proof summaries by the server._ Next, the server
moves to the last step of verifying the proof summaries _𝜎_ _𝑖_ = ( _𝑤_ _𝑖_ _[𝑜𝑢𝑡]_ _, 𝜆_ _𝑖_ )
for all clients not in C [∗] . Recall from the discussion in Sec. 4.1 that
this involves recovering the values _𝑤_ _𝑖_ _[𝑜𝑢𝑡]_ and _𝜆_ _𝑖_ from the shares
of _𝜎_ _𝑖_ and checking whether _𝑤_ _𝑖_ _[𝑜𝑢𝑡]_ = 1 and _𝜆_ _𝑖_ = 0 . However, we cannot simply use the naive share reconstruction algorithm from Sec.
4.1 since some of the shares might be incorrect (submitted by the


E FFeL: Ensuring Integrity For Federated Learning CCS ’22, November 7–11, 2022, Los Angeles, CA, USA



malicious clients). To address this issue, EIFFeL performs a _robust_
_reconstruction_ of the shares as follows. A naive strategy would be
sampling multiple subsets of _𝑚_ + 1 shares (each subset can emulate a SNIP setting), reconstructing the secret for each subset, and
taking the majority vote. However, we can do much better by exploiting the connections between Shamir’s secret shares and ReedSolomon error correcting codes (Sec. 4.1). Specifically, the Shamir’s
secret sharing scheme used by EIFFeL is a [ _𝑛_ −1 _,𝑚_ + 1 _,𝑛_ − _𝑚_ ] ReedSolomon code that can correct up to _𝑞_ errors and _𝑒_ erasures (message
dropouts) where 2 _𝑞_ + _𝑒_ _< 𝑛_ − _𝑚_ −1 . The server S can, therefore, use
SS.robustRecon(·) to reconstruct the secret when _𝑚_ _<_ ⌊ _[𝑛]_ [−] ~~3~~ [1] [⌋] [.]


After the robust reconstruction of the proof summaries, the server
S verifies them and updates the list C [∗] with _all_ malicious clients
with malformed updates. Specifically:
∀C _𝑖_ ∈C \ C [∗]
�SS.robustRecon ({( _𝑗,𝑤_ _𝑖𝑗_ _[𝑜𝑢𝑡]_ [)}] [C] _𝑗_ [∈C\{C] [∗] [∪C] _𝑖_ [}] [)][ ≠] [1][ ∨]

SS.robustRecon ({( _𝑗, 𝜆_ _𝑖𝑗_ )} C _𝑗_ ∈C\{C ∗ ∪C _𝑖_ } ) ≠ 0 �

=⇒C [∗] ←C [∗] ∪C _𝑖_ _._


Additionally, if a client C _𝑖_ withholds some of the shares of the proof
summaries for other clients, C _𝑖_ is marked as malicious as well by
the server. Thus, in addition to the malicious clients listed above,
the list C [∗] now has all clients that have either:

- failed the proof verification, _i.e._, provided malformed updates,
OR

- withheld shares of proof summaries of other clients (malicious
message dropout).
To conclude the round, the server publishes the updated list C [∗] on
the public bulletin B.


**Round 4 (Compute Aggregate).** This is the final round of EIFFeL where the aggregate of the well-formed updates is computed.
If a client C _𝑖_ is on C [∗] wrongfully, it can dispute its malicious status
by showing the other clients the transcript of the robust reconstruction from all the shares of _𝜎_ _𝑖_ (publicly available on bulletin B ). If
any client C _𝑖_ ∈C successfully raises a dispute, all clients abort
the protocol because they conclude that the server S has acted
maliciously by trying to withhold a verified well-formed update
from the aggregation. If no client raises a successful dispute, every
client C _𝑖_ ∈C \ C [∗] generates its share of the aggregate, ( _𝑖,_ U _𝑖_ ) with
U _𝑖_ = [�] C _𝑗_ ∈C\C [∗] _[𝑢]_ _𝑗𝑖_ [, and sends that share to the server] [ S] [. Note that,]
herein, C _𝑖_ uses its own share of the update, ( _𝑖,𝑢_ _𝑖𝑖_ ), as well.
The server recovers the aggregate U = [�] C _𝑖_ ∈C\C [∗] [U] _𝑗_ [using robust]
reconstruction: U ← SS.robustRecon({( _𝑖,_ U _𝑖_ )} C _𝑖_ ∈C\C ∗ ) .


**Discussion.** EIFFeL meets the design goals of Sec. 4.3 as follows.

_Flexibility of Integrity Checks._ SNIP supports arbitrary arithmetic
circuits for Valid(·) . The server S can choose a different Valid(·) for
every iteration (the protocol described above corresponds to a single
iteration of model training in FL). Additionally, S can hold multiple
Valid 1 (·) _,_ - · · _,_ Valid _𝑘_ (·) and want to check whether the client’s update passes them all. For this, we have Valid _𝑖_ (·) return zero (instead
of one) on success. If _𝑤_ _𝑖_ _[𝑜𝑢𝑡]_ is the value on the output wire of the
circuit Valid _𝑖_ (·), the server chooses random values ( _𝑙_ 1 _,_ - · · _,𝑙_ _𝑘_ ) ∈ F _[𝑘]_

and recovers the sum [�] _[𝑘]_ _𝑖_ =1 _[𝑙]_ _[𝑖]_ [·] _[ 𝑤]_ _𝑖_ _[𝑜𝑢𝑡]_ in Round 3. If any _𝑤_ _𝑖_ _[𝑜𝑢𝑡]_ = 0, then
the sum will be non-zero with high probability and S will reject.



_Compatibility with FL’s Infrastructure._ Current deployments of FL
involves a _single_ server who wants to train the global model. Hence,
as explained above, we design SNIP to be compatible with a single
server in EIFFeL. Solutions involving two or more non-colluding
servers are unrealistic for FL. For instance, currently the server can
be owned by Meta who wants to train privately on the data of its
user base. For a two-server model here, the second server has to be
owned by an independent party. Moreover, both the servers have to
do an equal amount of computation for model training (verification,
aggregation etc) since SNIP uses secret shares. This would make
sense only if _both_ the servers are interested in training the model.
For instance, if Meta and Google collaborate to train a model on
their joint user base which is an unrealistic scenario.


_Efficiency._ EIFFeL’s usage of SNIP as the underlying ZKP is made
from the efficiency point of view. SNIP is a light-weight ZKP system that is _specialized for the server-client settings_ resulting in good
performance. For instance, its performance is about three-orders
of magnitude better than that of zkSNARKs [ 28 ]. Instead of using ZKPs, one alternative is to use standard secure multi-party
computation (MPC) for the entire aggregation to directly compute
U _𝑣𝑎𝑙𝑖𝑑_ = [�] C _𝑖_ [Valid][(] _[𝑢]_ _𝑖_ [) ·] _[ 𝑢]_ _𝑖_ [. However, doing the entire aggregation]
under MPC would result in a massive circuit with _𝑂_ ( _𝑛𝑑_ ) multiplication gates where _𝑑_ is the data dimension. Multiplications are
costly for MPC and each gate requires a round of communication
in general making the above computation prohibitively costly. Extending the computation to the malicious threat model would be
even costlier. This is where SNIP proves to be advantageous: SNIP
enables the verifiers to check all the multiplication gates very efficiently (in a non-interactive fashion) with just one polynomial
identity test (Sec. 4.1).




CCS ’22, November 7–11, 2022, Los Angeles, CA, USA Roy Chowdhury et al.



**Computation** **Communication**


**Client** _𝑂_ ( _𝑚𝑛𝑑_ ) _𝑂_ ( _𝑚𝑛𝑑_ )
**Server** _𝑂_ [�] ( _𝑛_ + _𝑑_ ) _𝑛_ log [2] _𝑛_ log log _𝑛_ + _𝑚𝑑_ min( _𝑛,𝑚_ [2] ) [�] _𝑂_ [�] _𝑛_ [2] + _𝑚𝑑_ min( _𝑛,𝑚_ [2] ) [�]


**Table 1: Computational and communication complexity of EIF-**
**FeL for the server and an individual client.**



valid shares. Hence, at least _𝑛_ − _𝑚_ − 1 other honest clients C _𝐻_ \ C _𝑖_
will produce correct shares of the proof summary _𝜎_ _𝑖_ = ( _𝑤_ _𝑖_ _[𝑜𝑢𝑡]_ _, 𝜆_ _𝑖_ ) .
Using Fact 2, the server S is able to correctly reconstruct the value
of _𝜎_ _𝑖_ . Eq. 4 is now implied by the completeness property of SNIP.

                     

Lemma 3. _All updates accepted by EIFFeL are well-formed with prob-_
_ability_ 1 − negl( _𝜅_ ) _._





∀C _𝑖_ ∈C _,_ Pr
_EIFFeL_



� _Valid_ ( _𝑢_ _𝑖_ ) = 1 �� _Accept_ _𝑢_ _𝑖_ � = 1 − negl( _𝜅_ ) _._ (5)



Table 1 analyses the complexity of EIFFeL in terms of the number
of clients _𝑛_, number of malicious clients _𝑚_ and data dimension _𝑑_ .
We assume that |Valid| is of the order of _𝑂_ ( _𝑑_ ) . The total number of
one-way communication is 12 and 9 for the clients and the server,
respectively. A per-round analysis is presented in App. 11.2.


**5** **Security Analysis**


In this section, we formally analyze the security of EIFFeL.


Theorem 1. _For any public validation predicate_ _Valid_ (·) _that can be_
_expressed by an arithmetic circuit, EIFFeL is a SAVI protocol (Def. 1)_
_for_ |C _𝑀_ | _<_ ⌊ _[𝑛]_ [−] ~~3~~ [1] [⌋] _[and]_ [ C] _[Valid]_ [=][ C \ C] [∗] _[.]_


We present a proof sketch of the above theorem here; the formal
proof is in App. 11.5.


_Proof Sketch._ The proof relies on the following two facts.
**Fact 1.** _Any set of_ _𝑚_ _or less shares in EIFFeL reveals nothing about_
_the secret._

**Fact 2.** _A_ ( _𝑛,𝑚_ + 1 _,𝑛_ − _𝑚_ ) _Reed-Solomon error correcting code can_
_correctly construct the message with up to_ _𝑞_ _errors and_ _𝑒_ _erasures_
_(message dropout), where_ 2 _𝑞_ + _𝑒_ _< 𝑛_ − _𝑚_ + 1 _. In EIFFeL, we have_ _𝑞_ + _𝑒_ = _𝑚_
_where_ _𝑞_ _is the number of malicious clients that provide erroneous_
_shares and_ _𝑒_ _is the number of clients that withhold a message or are_
_barred from participation (_ i.e. _, are in_ C [∗] _)._


_Integrity._ We prove that EIFFeL satisfies the integrity constraint of
the SAVI protocol using the following three lemmas.


Lemma 2. _EIFFeL accepts the update of every honest client._


∀C _𝑖_ ∈C _𝐻_ : Pr (4)
_EIFFeL_ [[] _[Accept][ 𝑢]_ _[𝑖]_ []][ =][ 1] _[.]_


Proof. By definition, client C _𝑖_ ∈C _𝐻_ has well-formed inputs, that
is, Valid( _𝑢_ _𝑖_ ) = 1. Additionally, C _𝑖_, by virtue of being honest, submits



The proof relies on the fact that a client will be verified only if it
has submitted ≥ _𝑛_ − _𝑚_ − 1 valid shares (see App. 11.3).
Corollary 3.1. _EIFFeL rejects all malformed updates with probabil-_
_ity_ 1 − negl( _𝜅_ ) _._


Based on the above lemmas, at the end of Round 3, C \ C [∗] (set
of clients whose updates have been accepted) must contain _all_
honest clients C _𝐻_ . Additionally, it may contain some clients C _𝑖_
who have submitted well-formed updates with at least _𝑛_ − _𝑚_ − 1
valid shares for C _𝐻_, but may act maliciously for other steps of the
protocol (for instance, give incorrect shares of proof summary for
other clients or give incorrect shares of the final aggregate). This
is acceptable provided that EIFFeL is able to reconstruct the final
aggregate containing _only_ well-formed updates which is guaranteed
by the following lemma.
Lemma 4. _The aggregate_ U _must contain the updates of all honest_
_clients or the protocol is aborted._



C ⊆C \ {C¯ [∗] ∪C _𝐻_ } (6)


Proof. If the server S acts maliciously and publishes a list C [∗]

such that C [∗] ∩C _𝐻_ ≠ ∅, an honest client C _𝑖_ ∈C [∗] ∩C _𝐻_ publicly raises
a dispute. This is possible since all the shares of _𝜎_ _𝑖_ are publicly
logged on B . If the dispute is successful, all honest clients will
abort the protocol. Note that a malicious client with malformed
updates cannot force the protocol to abort in this way since it will
not be able to produce a successful transcript with high probability
(Lemma 3). If no clients raise a successful dispute, Eq. 6 follows
directly from Fact 2. C [¯] represents a set of malicious clients with
well-formed updates which corresponds to C Valid \ C _𝐻_ in Eq. 3. 

_Privacy._ The privacy constraint of SAVI states that nothing should be
revealed about a private update _𝑢_ _𝑖_ for an honest client C _𝑖_, except:

- _𝑢_ _𝑖_ passes the integrity check, _i.e._, Valid( _𝑢_ _𝑖_ ) = 1

- anything that can be learned from the aggregate of honest clients,
U _𝐻_ .
We prove that EIFFeL satisfies this privacy constraint with the help
of the following two helper lemmas.


Lemma 5. _In Rounds 1-3, for an honest client_ C _𝑖_ ∈C _𝐻_ _, EIFFeL reveals_
_nothing about 𝑢_ _𝑖_ _except Valid_ ( _𝑢_ _𝑖_ ) = 1 _._
The proof uses the fact that only _𝑚_ shares of C _𝑖_, which correspond
to the _𝑚_ malicious clients, can be revealed (see App. 11.4).


Lemma 6. _In Round_ 4 _, for an honest client_ C _𝑖_ ∈C _𝐻_ _, EIFFeL reveals_
_nothing about_ _𝑢_ _𝑖_ _except whatever can be learned from the aggregate._

Proof. In Round 4, from Lemma 4 and Fact 2, the information
revealed is either the aggregate or ⊥. 


U = U _𝐻_ +
∑︁



_𝑢_ _𝑖_ _where_ U _𝐻_ =

∑︁ ∑︁

C _𝑖_ ∈ C [¯] C _𝑖_ ∈C



_𝑢_ _𝑖_

C _𝑖_ ∈C _𝐻_


E FFeL: Ensuring Integrity For Federated Learning CCS ’22, November 7–11, 2022, Los Angeles, CA, USA


 - **Setup Phase.**

**–** All parties are given the security parameter _𝜅_, the number of clients _𝑛_ out of which at most _𝑚_ _<_ ⌊ _[𝑛]_ 3 [−][1] [⌋] [are malicious, honestly generated] _[ 𝑝𝑝]_ ←− $ KA.gen( _𝜅_ )

and a field F to be used for secret sharing. Server initializes lists Flag[ _𝑖_ ] = ∅ _,𝑖_ ∈[ _𝑛_ ] and C [∗] = ∅ .

 - **Round 1 (Announcing Public Information).**
_Client_ : Each client C _𝑖_

**–** Generates its key pair and announces the public key. ( _𝑝𝑘_ _𝑖_ _,𝑠𝑘_ _𝑖_ ) ←− $ KA.gen( _𝑝𝑝_ ), C _𝑖_ −−−→B _𝑝𝑘_ _𝑖_ .

_Server_ :

**–** Publishes the validation predicate Valid(·). S −−−−−−→B Valid(·)

 - **Round 2 (Generate and Distribute Proof).**
_Client_ : Each client _𝐶_ _𝑖_
**–** Computes _𝑛_ − 1 pairwise keys. ∀C _𝑗_ ∈C \ _𝑖_ _,𝑠𝑘_ _𝑖𝑗_ ← KA.agree( _𝑝𝑘_ _𝑗_ _,𝑠𝑘_ _𝑖_ )
**–** Generates proof _𝜋_ _𝑖_ = [�] _ℎ_ _𝑖_ _,_ ( _𝑎_ _𝑖_ _,𝑏_ _𝑖_ _,𝑐_ _𝑖_ ) [�] _,ℎ_ _𝑖_ ∈ F[ _𝑋_ ] _,_ ( _𝑎_ _𝑖_ _,𝑏_ _𝑖_ _,𝑐_ _𝑖_ ) ∈ F [3] _, 𝑎_ _𝑖_   - _𝑏_ _𝑖_ = _𝑐_ _𝑖_ for the statement Valid( _𝑢_ _𝑖_ ) = 1.

**–** Generates shares of the input _𝑢_ _𝑖_ ∈ F _[𝑑]_ . {(1 _,𝑢_ _𝑖_ 1 ) _,_   - · · _,_ ( _𝑛,𝑢_ _𝑖𝑛_ ) _,_ Ψ _𝑢_ _𝑖_ } ←− $ SS.share( _𝑢_ _𝑖_ _,_ [ _𝑛_ ] _,𝑚_ + 1)
**–** Generates shares of the proof _𝜋_ _𝑖_ . $ $
{(1 _,ℎ_ _𝑖_ 1 ) _,_            - · · _,_ ( _𝑛,ℎ_ _𝑖𝑛_ ) _,_ Ψ _ℎ_ _𝑖_ } ←− SS.share( _ℎ_ _𝑖_ _,_ [ _𝑛_ ] \ _𝑖,𝑚_ + 1) _,_ {(1 _,𝑎_ _𝑖_ 1 ) _,_            - · · _,_ ( _𝑛,𝑎_ _𝑖𝑛_ ) _,_ Ψ _𝑎_ _𝑖_ } ←− SS.share( _𝑎_ _𝑖_ _,_ [ _𝑛_ ] \ _𝑖,𝑚_ + 1)


$ $
{(1 _,𝑏_ _𝑖_ 1 ) _,_            - · · _,_ ( _𝑛,𝑏_ _𝑖𝑛_ ) _,_ Ψ _𝑏_ _𝑖_ } ←− SS.share( _𝑏_ _𝑖_ _,_ [ _𝑛_ ] \ _𝑖,𝑚_ + 1) _,_ {(1 _,𝑐_ _𝑖_ 1 ) _,_            - · · _,_ ( _𝑛,𝑐_ _𝑖𝑛_ ) _,_ Ψ _𝑐_ _𝑖_ } ←− SS.share( _𝑐_ _𝑖_ _,_ [ _𝑛_ ] \ _𝑖,𝑚_ + 1)

**–** Encrypts proof strings for all other clients. ∀C _𝑗_ ∈C \ _𝑖_ _,_ ( _𝑗,𝑢_ _𝑖𝑗_ ) ||( _𝑗, 𝜋_ _𝑖𝑗_ ) ←− $ AE.enc [�] _𝑠𝑘_ _𝑖𝑗_ _,_ ( _𝑗,𝑢_ _𝑖𝑗_ ) ||( _𝑗, 𝜋_ _𝑖𝑗_ ) [�] _, 𝜋_ _𝑖𝑗_ = _ℎ_ _𝑖𝑗_ || _𝑎_ _𝑖𝑗_ || _𝑏_ _𝑖𝑗_ || _𝑐_ _𝑖𝑗_ .

**–** Publishes check strings and the encrypted proof strings on the bulletin. ∀C _𝑗_ ∈C \ _𝑖_ _,_ C _𝑖_ −−−−−−−−−−−−−→B ( _𝑗,𝑢_ _𝑖𝑗_ )||( _𝑗,𝜋_ _𝑖𝑗_ ) ; C _𝑖_ −−−−−−−→B Ψ _𝑢𝑖_ _,_ Ψ _𝜋𝑖_

 - **Round 3 (Verify Proof)** .
(i) _Verifying validity of secret shares_ :
_Client_ : Each client C _𝑖_
**–** Downloads and decrypts proof strings for all other clients from the public bulletin. Flags a client in case their decryption fails.



∀C _𝑗_ ∈C \ _𝑖_ _,_ C _𝑖_



( _𝑖,𝑢_ _𝑗𝑖_ )||( _𝑖,𝜋_ _𝑗𝑖_ ) _,_ Ψ _𝑢𝑗_ _,_ Ψ _𝜋𝑗_
←−−−−−−−−−−−−−−−−−−−−B _,_ ( _𝑖,𝑢_ _𝑗𝑖_ ) ||( _𝑖, 𝜋_ _𝑗𝑖_ ) ← AE.dec [�] _𝑠𝑘_ _𝑖𝑗_ _,_ ( _𝑖,𝑢_ _𝑗𝑖_ ) ||( _𝑖, 𝜋_ _𝑗𝑖_ ) [�]


Flag C _𝑗_
⊥← AE.dec [�] _𝑠𝑘_ _𝑖𝑗_ _,_ ( _𝑖,𝑢_ _𝑗𝑖_ ) ||( _𝑖, 𝜋_ _𝑗𝑖_ ) [�] =⇒ _𝐶𝑙_ _𝑖_ −−−−−−→B



**–** Verifies the shares _𝑢_ _𝑗𝑖_ ( _𝜋_ _𝑗𝑖_ ) using checkstrings Ψ _𝑢_ _𝑗_ (Ψ _𝜋_ _𝑗_ ) and flags all clients with invalid shares. Flag C _𝑗_
_Server_ : ∀C _𝑗_ ∈C \ _𝑖_ _,_ 0 ← [�] SS.verify(( _𝑖,𝑢_ _𝑗𝑖_ ) _,_ Ψ _𝑢_ _𝑗_ ) ∧ SS.verify(( _𝑖, 𝜋_ _𝑗𝑖_ ) _,_ Ψ _𝜋_ _𝑗_ ) [�] =⇒C _𝑖_ −−−−−−→B
**–** If client C _𝑖_ flags client C _𝑗_, the server updates Flag[ _𝑗_ ] = Flag[ _𝑗_ ] ∪C _𝑖_
**–** Updates the list of malicious client C [∗] as follows:

   - Adds all clients who have flagged ≥ _𝑚_ + 1 other clients. ∀C _𝑖_ s. t. _𝑍_ = { _𝑗_ |C _𝑖_ ∈ Flag[ _𝑗_ ]} _,_ | _𝑍_ | ≥ _𝑚_ + 1 =⇒C [∗] ←C [∗] ∪C _𝑖_

   - Adds all clients with more than _𝑚_ + 1 flag reports. |Flag[ _𝑖_ ] | ≥ _𝑚_ + 1 =⇒C [∗] ←C [∗] ∪C _𝑖_

   - For clients with less flag reports, the server obtains the corresponding shares in the clear, verifies them and updates C [∗] accordingly. ∀C _𝑗_ s.t 1 ≤
|Flag[ _𝑗_ ] | ≤ _𝑚,_ ∀C _𝑖_ s.t. C _𝑖_ has flagged C _𝑗_

( _𝑖,𝑢_ _𝑗𝑖_ ) _,_ ( _𝑖,𝜋_ _𝑗𝑖_ )
−C _𝑗_ −−−−−−−−−−−−→B
− if [�] SS.verify(( _𝑖,𝑢_ _𝑗𝑖_ ) _,_ Ψ _𝑢_ _𝑗_ ) ∧ SS.verify(( _𝑖, 𝜋_ _𝑗𝑖_ ) _,_ Ψ _𝜋_ _𝑗_ ) [�] = 0 =⇒C [∗] ←C [∗] ∪C _𝑗_ _,_ otherwise, C _𝑖_ uses the verified shares to compute its proof

summary _𝜎_ _𝑗𝑖_
C [∗]
**–** Publishes C [∗] on the bulletin. S −−→B

(ii) _Generation of proof summaries by the clients._

_Server_ :

_𝑟_
**–** Server announces a random number _𝑟_ ∈ F. S −→B

_Client_ : Each client C _𝑖_ ∈C \ C [∗]

**–** Generates a summary _𝜎_ _𝑗𝑖_ of the proof string _𝜋_ _𝑗𝑖_ based on _𝑟_, ∀C _𝑗_ ∈C \ (C [∗] ∪C _𝑖_ ) _,_ C _𝑖_ ←−B _𝑟_ _, 𝜎_ _𝑗𝑖_ = [�] ( _𝑖, 𝑤_ _𝑜𝑢𝑡𝑗𝑖_ ) _,_ ( _𝑖, 𝜆_ _𝑗𝑖_ ) [�] _,_ C _𝑖_ −−−→B _𝜎_ _𝑗𝑖_
(iii) _Verification of proof summaries by the server._

_Server_ :

**–** Downloads and verifies the proof for all clients not on C [∗] via robust reconstruction of the digests and updates C [∗] accordingly.

_𝜎_ _𝑖𝑗_
∀C _𝑖_ ∈C \ C [∗] _,_ S ←−−−B _,_ [�] SS.robustRecon({( _𝑗, 𝑤_ _𝑖𝑗_ _[𝑜𝑢𝑡]_ ) } C _𝑗_ ∈C\(C ∗ ∪C _𝑖_ ) ) ≠ 1 ∨ SS.robustRecon({( _𝑗, 𝜆_ _𝑖𝑗_ ) } C _𝑗_ ∈C\(C ∗ ∪C _𝑖_ ) ) ≠ 0 [�] =⇒C [∗] ←C [∗] ∪C _𝑖_

C [∗]
**–** Publishes the updated list C [∗] on the bulletin. S −−→B

- **Round 4 (Compute Aggregate).**
_Client_ : Each client C _𝑖_
**–** If C _𝑖_ is on C [∗], C _𝑖_ raises a dispute by sending the transcript of the reconstruction of _𝜎_ _𝑖_ that shows _𝜆_ _𝑖_ = 0 ∧ _𝑤_ _[𝑜𝑢𝑡]_ _𝑗_ = 1 and aborts, OR

_𝜎_ _𝑖𝑗_ Transcript of SS.robustRecon({( _𝑗,𝜎_ _𝑖𝑗_ )} C _𝑗_ ∈C\(C∗∪C _𝑖_ ) )
∀C _𝑗_ ∈C \ _𝑖_ _,_ C _𝑖_ ←−−−B _,_ C _𝑖_ −−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−→B
**–** Aborts protocol if it sees any other client on C [∗] successfully raise a dispute, OR

**–** If no client has raised a dispute and C _𝑖_ is not on C [∗], sends the aggregate of the shares of clients in C \ C [∗] to the server. U _𝑖_ = [�] _𝑢_ _𝑗𝑖_ −−→S U _𝑖_
_Server_ : C _𝑗_ ∈C\C [∗] _[,]_ [ C] _[𝑖]_
**–** Reconstructs the final aggregate. U ← SS.robustRecon({( _𝑖,_ U _𝑖_ ) } C _𝑖_ ∈C\C ∗ )


**Figure 4: EIFFeL: Description of the secure aggregation with verified inputs protocol.**


CCS ’22, November 7–11, 2022, Los Angeles, CA, USA Roy Chowdhury et al.



**6** **EIFFeL Optimizations**


**6.1** **Probabilistic Reconstruction**


The Gao’s decoding algorithm alongside the use of verifiable secret
sharing guarantees that the correct secret will be recovered (with
probability one). However, we can improve performance at the cost
of a small probability of failure.


**Verifying Secret Shares.** As discussed in Sec. 11.2, verifying the
validity of the secret shares is the dominating cost for client-side
computation. To reduce this cost, we propose an optimization
where the validation of the shares corresponding to the proof
_𝜋_ _𝑖_ = [�] _ℎ_ _𝑖_ _,_ ( _𝑎_ _𝑖_ _,𝑏_ _𝑖_ _,𝑐_ _𝑖_ ) [�] can be eliminated. Specifically, we propose the
following changes to Round 3:


- Each client C _𝑖_ skips verifying the validity of the shares ( _𝑖, 𝜋_ _𝑗𝑖_ )
for C _𝑗_ ∈C \ _𝑖_ .

- Let _𝑒_ = |C [∗] | . The server S samples two sets of clients _𝑃_ 1 _, 𝑃_ 2 from

C \ {C _𝑖_ ∪C [∗] } of size at least 3 _𝑚_ − 2 _𝑒_ + 1 ( _𝑃_ 1 _, 𝑃_ 2 can be overlapping) and performs Gao’s decoding on both the sets to obtain
polynomials _𝑝_ 1 and _𝑝_ 2 . The server accepts the _𝑤_ _𝑖_ _[𝑜𝑢𝑡]_ ( _𝜆_ _𝑖_ ) only
iff _𝑝_ 1 = _𝑝_ 2 and _𝑝_ 1 (0) = _𝑝_ 1 (0) = 1( _𝑝_ 1 (0) = _𝑝_ 1 (0) = 0) . The cost of this
step is _𝑂_ ( _𝑛_ [2] log [2] _𝑛_ log log _𝑛_ ) which is less than verifying the shares
of _𝜋_ _𝑖_ when _𝑚_ _< 𝑛_ ≪ _𝑑_ (improves runtime by 2 _._ 3×, see Table 2).


Note that a [ _𝑛,𝑘,𝑛_ − _𝑘_ + 1] Reed-Solomon error correcting code can
correct up to ⌊ _[𝑛]_ [−] ~~2~~ _[𝑘]_ [−] _[𝑙]_ ⌋ errors with _𝑙_ erasures. Thus, with _𝑚_ − _𝑒_ mali
cious clients, only 3 _𝑚_ −2 _𝑒_ +1 shares are sufficient to correctly reconstruct the secret for honest clients. Since, the random sets _𝑃_ 1 and
_𝑃_ 2 are not known, a malicious client with more than _𝑚_ − _𝑒_ invalid
shares can cheat only with probability at most [1] /( [3] _[𝑚]_ _𝑛_ [−] − [2] _𝑒_ _[𝑒]_ [+][2] [)] [. We cannot]
extend this technique for the secret shares of the update _𝑢_, because,
unlike the value of the digests ( _𝑤_ _[𝑜𝑢𝑡]_ = 1 _, 𝜆_ = 0), the final aggregate is
unknown and needs to be reconstructed from the shares.


_Improvement._ Eliminates verification of check strings for the proof
_𝜋_ _𝑖_ which reduces time by 2 _._ 3× (Table 2).
_Cost._ Additional [1] /( [3] _[𝑚]_ _𝑛_ [−] − [2] _𝑒_ _[𝑒]_ [+][2] [)] [ probability of failure where] _[ 𝑒]_ [=][ |] _[𝐶]_ [∗] [|] [ .]


**Robust Reconstruction.** In case _𝑚_ ≤ [√] ~~_𝑛_~~ − 2, the robust reconstruction mechanism can be optimized as follows. Let _𝑞_ = _𝑚_ −|C [∗] |
be the number of malicious clients that remain undetected. The
server S partitions the set of clients in C \ C [∗] into at least _𝑞_ + 2
disjoint partitions, _𝑃_ = { _𝑃_ 1 _,_ - · · _, 𝑃_ _𝑞_ +2 } each of size _𝑚_ + 1. Let
_𝑝_ _𝑗_ ( _𝑥_ ) = _𝑐_ _𝑗,_ 0 + _𝑐_ _𝑗,_ 1 _𝑥_ + _𝑐_ _𝑗,_ 2 _𝑥_ [2] + · · · + _𝑐_ _𝑗,𝑚_ _𝑥_ _[𝑚]_ represent the polynomial corresponding to the _𝑚_ + 1 shares of partition _𝑃_ _𝑗_ . Recall
that recovering just _𝑝_ _𝑗_ ( 0 ) = _𝑐_ _𝑗,_ 0 suffices for a typical Shamir secret
share reconstruction. However, now, the server S recovers the entire polynomial _𝑝_ _𝑗_, _i.e._, all of its coefficients { _𝑐_ _𝑗,_ 0 _,𝑐_ _𝑗,_ 1 _,_ - · · _,𝑐_ _𝑗,𝑞_ } for
all _𝑞_ + 2 partitions. Based on the pigeon hole principle, it can be
argued that at least two of the partitions ( _𝑃_ _𝑙_ _, 𝑃_ _𝑘_ ∈ _𝑃_ ) will consist of
_honest_ clients only. Hence, we must have at least two polynomials
_𝑝_ _𝑙_ and _𝑝_ _𝑘_ that match and the value of the secret is their constant
coefficient _𝑝_ _𝑙_ ( 0 ) . Note that the above mentioned optimization of
skipping verifying the shares of the proof can be applied here as
well. A malicious client can cheat ( _i.e._, make the server S accept
even when _𝑤_ _𝑖_ _[𝑜𝑢𝑡]_ ≠ 1 ∨ _𝜆_ _𝑖_ ≠ 0 or reject the proof for an honest client)
only if they can manipulate the shares of at least two partitions
which must contain at least 2( _𝑚_ + 1) − _𝑞_ honest clients. Since the



random partition _𝑃_ is not known to the clients, this can happen
only with probability [1] /( [2][(] _𝑛_ _[𝑚]_ − _𝑚_ [+][1] − [)−] 1 _[𝑞]_ [)][.]


_Improvement._ Reduces the number of polynomial interpolations.
_Cost._ Additional [1] /( [2][(] _𝑛_ _[𝑚]_ − _𝑚_ [+][1] − [)−] 1 _[𝑞]_ [)] [ probability of failure where] _[ 𝑞]_ [=] _[ 𝑚]_ [−|C] [∗] [|] [ .]

**6.2** **Crypto-Engineering Optimizations**


**Equality Checks.** The equality operator = is relatively complicated to implement in an arithmetic circuit. To circumvent this issue,
we replace any validation check of the form Φ( _𝑢_ ) = _𝑐_ 1 ∨ Φ( _𝑢_ ) = _𝑐_ 2 ∨· · ·
∨Φ( _𝑢_ ) = _𝑐_ _𝑘_ in the output nodes of Valid(·), where Φ(·) is some arithmetic function, by an output of the form (Φ( _𝑢_ ) − _𝑐_ 1 ) × · · · × (Φ( _𝑢_ ) − _𝑐_ _𝑘_ ) .
Recall that in EIFFeL, the honest clients have well-formed inputs
that satisfy Valid(·) by definition. Hence, this optimization does not
violate the privacy of honest, which is our security goal.


_Improvement._ Reduces the circuit size |Valid|.
_Cost._ No cost.


**Proof Summary Computation.** In addition to being a linear secret sharing scheme, Shamir’s scheme is also multiplicative: given
the shares of two secrets ( _𝑖,𝑧_ _𝑖_ ) and ( _𝑖, 𝑣_ _𝑖_ ), a party can locally compute ( _𝑖,𝑠_ _𝑖_ ) with _𝑠_ = _𝑧_ - _𝑣_ . However, if the original shares correspond
to a polynomial of degree _𝑡_, the new shares represent a polynomial
of degree 2 _𝑡_ . Hence, we do not rely on this property for the multiplication gates of Valid(·) as it would support only limited number
of multiplications. However, if _𝑚_ _<_ _[𝑛]_ [−] ~~4~~ [1] [, we can still leverage the]

multiplicative property to generate shares of the random digest
_𝜆_ _𝑖_ = _𝑓_ _𝑖_ ( _𝑟_ ) · _𝑔_ _𝑖_ ( _𝑟_ ) = _ℎ_ _𝑖_ ( _𝑟_ ) locally (instead of using Beaver’s triples).


_Improvement._ Saves a round of communication and reduces the
number of robust reconstructions for _𝜆_ _𝑖_ from three to just one
(details in App. 11.1).
_Cost._ No cost.


**Random Projection** . As shown in Table 1, both communication
and computation grows linearly with the data dimension _𝑑_ . Hence,
we rely on the random projection [ 66 ] technique for reducing the
dimension of the updates. Specifically, we use the fast random
projection using Walsh-Hadamard transforms [4].


_Improvement._ Reduces the data dimension which helps both computation and communication cost.
_Cost._ Empirical evaluation (Sec. 7.2) shows that the efficacy of
Valid(·) is still preserved.


**7** **Experimental Evaluation**


**7.1** **Performance Evaluation.**


In this section, we analyze the performance of EIFFeL.


**Configuration.** We run experiments on two Amazon EC2 c5.9large
instances with Intel Xeon Platinum 8000 processors. To emulate
server-client communication, we use two instances in the US East
(Ohio) and US West (Oregon) regions, with a round trip time of
21 ms. We implemented EIFFeL in Python and C++ using NTL library [ 1 ]. We use AES-GCM for encryption, a 56-bit prime field
F and probabilistic quantization [ 47 ]. For key agreement, we use
elliptic curve Diffie-Hellman [ 32 ] over the NIST P-256 curve. Unless otherwise specified, the default settings are _𝑑_ = 1 _𝐾_, _𝑛_ = 100,
_𝑚_ = 10% and |Valid(·)| ≈ 4 _𝑑_ . We report the mean of 10 runs for each


E FFeL: Ensuring Integrity For Federated Learning CCS ’22, November 7–11, 2022, Los Angeles, CA, USA





















**Figure 5: Computation cost analysis of EIFFeL. The left two plots show the runtime of a single client client in milliseconds as a function of:**
**(left) the number of clients** _𝑛_ **and (right) dimensionality of the updates** _𝑑_ **. The right two plots show the runtime of the server as a function of**
**the same variables. The results demonstrate that performance decays quadratically in** _𝑛_ **, and linearly in** _𝑑_ **.**
















|100 (MB) 5% malicious 10%    "|5% malicious 10%    "|s clients|Col4|
|---|---|---|---|
|~~50~~<br>~~100~~<br>Numb<br>0<br>20<br>40<br>60<br>80<br><br><br>15%       "<br>~~20%       "~~|15%       "<br>~~20%       "~~|||
|~~50~~<br>~~100~~<br>Numb<br>0<br>20<br>40<br>60<br>80<br><br><br>15%       "<br>~~20%       "~~||||
|~~50~~<br>~~100~~<br>Numb<br>0<br>20<br>40<br>60<br>80<br><br><br>15%       "<br>~~20%       "~~||~~150~~<br>~~200~~<br>~~250~~<br>er of Clients|~~150~~<br>~~200~~<br>~~250~~<br>er of Clients|


|200 (MB)|Col2|Col3|Col4|
|---|---|---|---|
|~~1,000~~<br>~~5,000~~<br>~~10,00~~<br>Data Dimension<br>0<br>50<br>100<br>150<br>||||
|~~1,000~~<br>~~5,000~~<br>~~10,00~~<br>Data Dimension<br>0<br>50<br>100<br>150<br>||||


|100 (MB)|Col2|Server|r|
|---|---|---|---|
|~~50~~<br>~~100~~<br>N<br>0<br>25<br>50<br>75<br>||||
|~~50~~<br>~~100~~<br>N<br>0<br>25<br>50<br>75<br>||||
|~~50~~<br>~~100~~<br>N<br>0<br>25<br>50<br>75<br>||~~150~~<br>mber of|~~200~~<br>~~250~~<br>lients|


|150 180 (MB) Server|Serv|ver|Col4|
|---|---|---|---|
|~~1,000~~<br>~~5,000~~<br>~~10,00~~<br>Data Dimension<br>0<br>30<br>60<br>90<br>120<br>150<br>||||
|~~1,000~~<br>~~5,000~~<br>~~10,00~~<br>Data Dimension<br>0<br>30<br>60<br>90<br>120<br>150<br>||||



**Figure 6: Communication cost analysis of EIFFeL. The left two plots show the amount of communication (in MB) for each client as a function**
**of: (left) the number of clients** _𝑛_ **and (right) dimensionality of the updates** _𝑑_ **. The right two plots show the the amount of communication (in**
**MB) for the server as a function of the same variables. The results show communication increases quadratically in** _𝑛_ **, and linearly in** _𝑑_ **.**



experiment. The rejection probability ( _𝑛𝑒𝑔𝑙_ ( _𝜅_ )) is dominated by
2M−2 / |F | (soundness error of SNIP, Sec. 4.1). M _<_ 4 _𝑑_ _<_ 40 _𝐾_ in our
evaluation so the failure probability is of the order of _𝑂_ (10 [−][12] ).


**Computation Costs.** Fig. 5 presents EIFFeL’s runtime. We vary
the number of malicious clients between 5%-20% of the number of

clients. We observe that per-client runtime of EIFFeL is low: it is
1 _._ 3 _𝑠_ if _𝑚_ = 10%, _𝑑_ = 1 _𝐾_, and _𝑛_ = 100. The runtime scales quadratically in _𝑛_ because a client has _𝑂_ ( _𝑚𝑛𝑑_ ) computation complexity (see
Table 1) and _𝑚_ is a linear function of _𝑛_ . As expected, the runtime
increases linearly with _𝑑_ . A client takes around 11 _𝑠_ when _𝑑_ = 10 _𝐾_,
_𝑛_ = 100, and _𝑚_ = 10%. The runtime for the server is also low: the
server completes its computation in about 1 _𝑠_ for _𝑛_ = 100, _𝑑_ = 1 _𝐾_,
and _𝑚_ = 10%. The server’s runtime also scales quadratically in _𝑛_
due to the _𝑂_ ( _𝑚𝑛𝑑_ ) computation complexity (Table 1). The runtime
increases linearly with _𝑑_ .


In Fig. 7, we break down the runtime per round. We observe that:
Round 1 (announcing public information) incurs negligible cost
for both clients and the server; and Round 3 (verify proof) is the
costliest round for both clients and the server where the dominating
cost is verifying the validity of the shares (Sec. 11.2). Note that the
server has no runtime cost for Round 2 since the proof generation
only involves clients.


Table 2 presents our end-to-end performance which contains the
runtimes of a client, the server and the communication latencies.
For instance, the end-to-end runtime for _𝑛_ = 100, _𝑑_ = 1 _𝐾_ and _𝑚_ = 10%
is ∼ 2 _._ 4 _𝑠_ . We also present the impact of one of our key optimizations – eliminating the verification of the secrets shares of the proof
– which cuts down the costliest step in EIFFeL and improves the
performance by 2 _._ 3 × . Additionally, we compare EIFFeL’s performance with BREA [80], which is a Byzantine-robust secure aggregator. EIFFeL differs from BREA in two key ways: ( 1 ) EIFFeL is a
general framework for per-client update integrity checks whereas



**Figure 7: Computation cost per round in EIFFeL.**

BREA implements the multi-Krum aggregation algorithm [ 15 ] that
considers the entire dataset to determine the malicious updates
(computes all the pairwise distances between the clients and then,
detects the outliers), and ( 2 ) BREA has an additional privacy leakage as it reveals the values of all the pairwise distances between
clients. Nevertheless, we choose BREA as our baseline because, to
the best of our knowledge, this is the only prior work that: ( 1 )
detects and removes malformed updates, and ( 2 ) works in the malicious threat model with ( 3 ) a single server (see Table 3, Sec. 9).
We observe that EIFFeL outperforms BREA and that the improvement increases with _𝑛_ . For instance, for _𝑛_ = 250, EIFFeL is 18 _._ 5 ×
more performant than BREA. This is due to BREA’s complexity
of _𝑂_ ( _𝑛_ [3] log [2] _𝑛_ log log _𝑛_ + _𝑚𝑛𝑑_ ), where the _𝑂_ ( _𝑛_ [3] ) factor is due to each
client partaking in the computation of the _𝑂_ ( _𝑛_ [2] ) pairwise distances.


**Communication Cost.** Fig. 6 depicts the total data transferred by



**Improvement over**


**# Clients** ( _𝑛_ ) **Time** (ms) Unoptimized EIFFeL BREA [80]


50 1,072 2.3× 2.5×

100 2,367 2.3× 5.2×

150 4,326 2.3× 7.8×

200 6,996 2.3× 12.8×

250 10,389 2.3× 18.5×


**Table 2: End-to-end time for a single iteration of EIFFeL with** _𝑑_ = 1000
**and** _𝑚_ = 10% **malicious clients, as a function of the number of clients,**
_𝑛_ **. We also compare it with a variant of EIFFeL without optimiza-**
**tions, and with BREA [80].**










CCS ’22, November 7–11, 2022, Los Angeles, CA, USA Roy Chowdhury et al.







**(a) MNIST: Sign flip attack with norm**
**ball validation predicate (defense).**


**(e)** **MNIST:** **Min-Max** **attack** **with**
**Zeno++ validation predicate.**



|100|Col2|
|---|---|
|~~1~~<br>40<br>60<br>80<br><br>Test Accuracy||
|~~1~~<br>40<br>60<br>80<br><br>Test Accuracy|~~00~~<br>~~200~~<br>~~300~~<br>~~400~~<br>~~500~~<br>Number of Iterations|


**(b) MNIST: Scaling attack and cosine**
**similarity validation predicate.**

|80<br>Accuracy<br>60<br>40 Test<br>20<br>100|Col2|
|---|---|
|~~100~~<br><br>20<br>40<br>60<br>80<br>Test Accuracy|~~300~~<br>~~500~~<br>~~700~~<br>~~900~~<br>umber of Iterations|



**(f) CIFAR-10: Min-Sum attack with co-**
**sine similarity validation predicate.**



|80<br>Accuracy<br>60<br>40<br>Test<br>20<br>100 200<br>Number|Col2|
|---|---|
|~~100~~<br>~~200~~<br>Number<br>20<br>40<br>60<br>80<br>Test Accuracy|~~300~~<br>~~400~~<br>~~500~~<br> of Iterations|


**(c) FMNIST: Additive noise attack with**
**Zeno++ validation predicate.**


|80<br>Accuracy<br>60<br>40<br>Test<br>20<br>1|Col2|
|---|---|
|~~1~~<br>20<br>40<br>60<br>80<br>Test Accuracy|~~00~~<br>~~200~~<br>~~300~~<br>~~400~~<br>~~500~~<br>Number of Iterations|



|100<br>80<br>Accuracy<br>60<br>40<br>Test<br>20<br>0<br>100 20<br>Numbe|Main task<br>Backdoor task|
|---|---|
|~~100~~<br>~~20~~<br>Numbe<br>0<br>20<br>40<br>60<br>80<br>100<br>Test Accuracy|~~0~~<br>~~300~~<br>~~400~~<br>~~500~~<br> of Iterations|


**(g) EMNIST: Backdoor Attack-1 with**
**norm bound validation predicate.**







**(d) FMNIST: Sign flip attack with norm**
**ball validation predicate.**

|100<br>80<br>Accuracy<br>60<br>40<br>Test<br>20<br>0<br>100|Col2|
|---|---|
|~~100~~<br>0<br>20<br>40<br>60<br>80<br>100<br>Test Accuracy|~~300~~<br>~~500~~<br>~~700~~<br>~~900~~<br>Number of Iterations|



**(h) CIFAR-10: Backdoor Attack-2 with**
**norm bound validation predicate.**



**Figure 8: Accuracy analysis of EIFFeL. Test accuracy is shown as a function of the FL iteration for different datasets and attacks.**



a client and the server. The communication complexity is _𝑂_ ( _𝑚𝑛𝑑_ )
for a single client and for the server. Hence, the total communication increases quadratically with _𝑛_ and linearly with _𝑑_, respectively.
We observe that EIFFeL has acceptable communication cost. For
instance, the total data consumed by a client is 132MB for the configuration _𝑛_ = 100 _,𝑑_ = 10 _𝐾,𝑚_ = 10% . This is equivalent to streaming
a full-HD video for 26 _𝑠_ [ 2 ]. Since most clients partake in FL training
iterations infrequently, this communication is acceptable.


**Note.** Recall, we assume the size of the validation predicate to be
|Valid| = _𝑂_ ( _𝑑_ ) since Valid(·) defines a function on the input which is
_𝑑_ -dimensional. This assumption is validated by the state-of-the-art
predicates tested in Sec. 7.2. The above experiments use |Valid| ≈ 4 _𝑑_ .
Hence, the overall complexity (App. 11.2) is dominated by the
_𝑂_ ( _𝑚𝑛𝑑_ ) term and does not depend on the validation predicate.


**7.2** **Integrity Guarantee Evaluation**


In this section, we evaluate EIFFeL’s efficacy in ensuring update
integrity on real-world datasets.
**Datasets.** We evaluate EIFFeL on three image datasets:

- _MNIST_ [ 51 ] is a digit classification dataset of 60 _𝐾_ training images
and 10 _𝐾_ test images with ten classes.

- _EMNIST_ [ 27 ] is a writer-annotated handwritten digit classification dataset with ∼ 340 _𝐾_ training and ∼ 40 _𝐾_ testing images.

- _FMNIST_ [ 96 ] is identical to MNIST in terms number of classes,
and number of training and test images.

- _CIFAR-10_ [ 49 ] contains RGB images with ten object classes. It
has 50 _𝐾_ training and 10 _𝐾_ test images.

**Models.** We test EIFFeL with three classification models:

- _LeNet-5_ [ 50 ] has five layers and 60 _𝐾_ parameters, and is used to
experiment on MNIST and EMNIST.

- For FMNIST, we use a five-layer convolutional network with 70 _𝐾_
parameters and a similar architecture as LeNet-5.

- We use _ResNet-20_ [ 40 ] with 20 layers and 273 _𝐾_ parameters for
CIFAR-10.



**Validation Predicates.** To demonstrate the flexibility of EIFFeL,
we evaluate four validations predicates, which represent the current
_state-of-the-art_ defenses against data poisoning, as follows:

- _Norm Bound_ [ 85 ]. This method checks whether the _ℓ_ 2 -norm of
a client update is bounded: Valid( _𝑢_ ) = I[∥ _𝑢_ ∥ 2 _< 𝜌_ ] where I[·] is
the indicator function and the threshold _𝜌_ is computed from the
public dataset D _𝑃_ .

- _Norm Ball_ [ 83 ]. This method checks whether a client update is
within a spherical radius from _𝑣_ which is the gradient update computed from the clean public dataset D _𝑃_ : Valid( _𝑢_ ) = I�∥ _𝑢_ − _𝑣_ ∥ 2 ≤ _𝜌_ �

where radius _𝜌_ is also computed from D _𝑃_ .

- _Zeno++_ [ 91 ] compares the client update with a loss gradient _𝑣_ that
is computed on the public dataset D _𝑃_ : Valid( _𝑢_ ) = I[ _𝛾_ ⟨ _𝑣,𝑢_ ⟩− _𝜌_ || _𝑢_ || 2
≥− _𝛾𝜖_ ] where _𝛾_, _𝜌_ and _𝜖_ are threshold parameters also computed
from D _𝑃_ and _𝑢_ is _ℓ_ 2 -normalized to have the same norm as _𝑣_ .

- _Cosine Similarity_ [ 5, 24 ]. This method compares the cosine similarity between the client update _𝑢_ and the global model update of
the last iteration _𝑢_ [′] : Valid( _𝑢_ ) = I� ∥ _𝑢_ ⟨∥ _𝑢_ 2 _,_ ∥ _𝑢𝑢_ [′] ⟩ ~~[′]~~ ∥ 2 _[<][ 𝜌]_ � where _𝜌_ is computed from D _𝑃_ and _𝑢_ is _ℓ_ 2 -normalized to match norm of _𝑢_ [′] .
**Poisoning Attacks.** To test the efficacy of EIFFeL’s implementations of the four validation predicates introduced above, we test it
against seven poisoning attacks:

- _Sign Flip Attack_ [ 31 ]. In this attack, the malicious clients flip the
sign of their local update: ˆ _𝑢_ = − _𝑐_  - _𝑢,𝑐_ ∈ R + .

- _Scaling Attack_ [ 10 ] scales a local update to increase its influence
on the global update: ˆ _𝑢_ = _𝑐_  - _𝑢,𝑐_ ∈ R + .

- _Additive Noise Attack_ [ 53 ] adds Gaussian noise to the local update:

_𝑢_ ˆ = _𝑢_ + _𝜂,𝜂_ ∼N ( _𝜎, 𝜇_ ) .

- _Min-Max Attack_ [ 77 ] sets all the malicious updates to be: argmax _𝛾_



max _𝑖_ ∈[ _𝑛_ ] || _𝑢_ ˆ − _𝑢_ _𝑖_ || 2 ≤ max _𝑖,𝑗_ ∈[ _𝑛_ ] || _𝑢_ _𝑖_ − _𝑢_ _𝑗_ || 2 ; ˆ _𝑢_ = ~~_𝑛_~~ [1] � _𝑛𝑖_ =1 _[𝑢]_ _[𝑖]_ [+] _[ 𝛾]_ [·] _[ 𝑢]_ _[𝑝]_ [,]

where _𝑢_ _[𝑝]_ is a dataset optimized perturbation vector. Here, the
adversary is assumed to have access to the benign (well-formed)
updates of _all_ clients. This attack finds the malicious gradient
whose maximum distance from a benign gradient is less than the
maximum distance between any two benign gradient.


E FFeL: Ensuring Integrity For Federated Learning CCS ’22, November 7–11, 2022, Los Angeles, CA, USA




- _Min-Sum Attack_ [ 77 ] sets all the malicious updates to be: argmax _𝛾_
� _𝑖_ ∈[ _𝑛_ ] [||] _[𝑢]_ [ ˆ] [−] _[𝑢]_ _𝑖_ [||] 2 [≤] [max] _𝑖_ ∈[ _𝑛_ ] � _𝑗_ ∈[ _𝑛_ ] [||] _[𝑢]_ _𝑖_ [−] _[𝑢]_ _𝑗_ [||] 2 [; ˆ] _[𝑢]_ [=] ~~_𝑛_~~ [1] � _𝑛𝑖_ =1 _[𝑢]_ _[𝑖]_ [+] _[ 𝛾]_ [·] _[ 𝑢]_ _[𝑝]_ [,]

where _𝑢_ _[𝑝]_ is a dataset optimized perturbation vector. Here, the
adversary is assumed to have access to the benign updates of _all_
clients. This attack finds the malicious gradient such that the sum
of its distances from all the other gradients is less than the sum
of distances of any benign gradient from other benign gradients.

- _Backdoor Attack-1_ [85] classifies the digit seven as the digit one
for EMNIST.

- _Backdoor Attack-2_ [ 5 ] classifies images of green cars as birds for
CIFAR-10.


**Configuration.** We use the same configuration as before. We implement the image-classification models in PyTorch. We randomly
select 10K samples from each training set as the public dataset D _𝑃_
and train on the remaining samples. EMNIST is collected from 3383
clients with ∼ 100 images per client. For all other datasets, the
training set is divided into 5K subsets to create the local dataset
for each client. For each training iteration, we sample the required
number of data subsets out of these 5K subsets.


**Results.** Fig. 8 shows the accuracy of different image-classification
models in EIFFeL. We set _𝑛_ = 100 and _𝑚_ = 10%, and use random
projection to project the updates to a dimension _𝑑_ of 1K (MNIST,
EMNIST), 5K (FMNIST), or 10K (CIFAR-10). For the two backdoor
attacks, we consider _𝑚_ = 5%. Our experiment assesses how the
random projection affects the efficacy of the integrity checks. We
observe that for MNIST (Figs. 8a, 8b and 8e), EMNIST (Fig. 8g) and
FMNIST (Fig. 8c and 8d), EIFFeL achieves performance comparable
to a baseline that applies the defense (validation predicate) on the
plaintext. In most cases, the defenses retain their efficacy even
after random projection. This is because they rely on computing
inner products and norms of the update; these operations preserve
their relative values after the projection with high probability [ 66 ].

∼
We observe a drop in accuracy ( 7%) on CIFAR-10 (Figs. 8f and
8h) as updates for ResNet-20 with 273K parameters are projected
to 10K. The end-to-end per-iteration time ( _𝑚_ = 10% ) for MNIST,
EMNIST, FMNIST, and CIFAR-10 is 2 _._ 4 _𝑠_ (Table 2), 2 _._ 4 _𝑠_, 10 _._ 7 _𝑠_, and
20 _._ 5 _𝑠_, respectively. The associated communication costs for the
client are 13 _._ 3MB, 13 _._ 3MB, 65 _._ 8MB, and 132MB (Fig. 6). Additional
evaluation results are presented in Fig. 9 (App. 11.6).


**8** **Discussion**


In this section, we discuss possible avenues for future research
(additional discussion in App. 11.7).



**Handling Higher Fraction of Malicious Clients.** For ⌊ _[𝑛]_ [−] ~~3~~ [1] [⌋] _[<][ 𝑚]_

_<_ ⌊ _[𝑛]_ [−] ~~2~~ [1] [⌋] [(honest majority), the current implementation of][ EIFFeL][ can]

detect but not remove malformed inputs (Gao’s decoding algorithm
returns ⊥ if _𝑚_ _>_ ⌊ _[𝑛]_ [−] ~~3~~ [1] [⌋] [). Robust reconstruction in this case could be]

done via Guruswami-Sudan list decoder [ 59 ]. We do not do so in
EIFFeL because the reconstruction might fail sometimes.


**Handling Client Dropouts.** In practice, clients might have only
sporadic access to connectivity and so, the protocol must be robust
to clients dropping out. EIFFeL can already accommodate malicious
client dropping out – it is straightforward to extend this for the
case of honest clients as well.



**Identifying All Malicious Clients.** Currently, EIFFeL identifies
a partial list of malicious clients. To detect all malicious clients,
one can use: ( 1 ) PVSS to identify all clients who have submitted
at least one invalid share, and ( 2 ) decoding algorithms such as
Berlekamp-Welch [ 13 ] that can detect the location of the errors
from the reconstruction. We do not use them in EIFFeL as they have
higher computation cost.


**Reducing Client’s Computation.** Currently, verifying the validity of the secret shares is the dominant cost for clients. This task
can be offloaded to the server S by using a publicly verifiable secret
sharing scheme (PVSS) [ 73, 82, 86 ] where the validity of a secret
share can be verified by any party. However, typically PVSS employs public key cryptography (which is costlier than symmetric
cryptography) which might increase the end-to-end running time.


**Additional Defense Strategies.** EIFFeL supports any defense strategy that can be expressed as a per-update anomaly detection mechanism (captured via the public validation predicate Valid(·) ). A recent
line of work [ 43, 52, 69, 71, 87, 89 ] proposes a complimentary style
of defense which involves inspecting the final aggregate and the
resulting model. For instance, Sparsefed [ 69 ] is a state-of-the-art
backdoor defense where the server selects the top _𝑘_ dimensions
of the final aggregate and updates the model only along those
dimensions (others are set to zero). CRFL [ 89 ] provides certified
robustness against backdoor attacks by clipping and perturbing the
final aggregate and performing parameter smoothing on the global
model during testing. Jia et al. [ 43 ] propose an ensemble learning
mechanism where the server learns multiple global models on randomly selected subset of clients and takes majority vote among
the global models for test-time prediction. Such defenses can be
immediately integrated with EIFFeL since the server has access to
the final aggregate and the updated global model in the clear.


**Towards poly-logarithmic complexity.** Currently, dominant term
in the complexity is _𝑂_ ( _𝑚𝑛𝑑_ ) which results in a _𝑂_ ( _𝑛_ [2] ) dependence
on _𝑛_ (since we consider _𝑚_ is a fraction of _𝑛_ ). This can be reduced to
_𝑂_ ( _𝑛_ log [2] _𝑛𝑑_ ) by using the techniques from [ 9 ]. A detailed discussion
is presented in App. 11.7.


**9** **Related Work**


**Table 3: Comparison of EIFFeL with Related Work**


Work Malicious Single Removes Arbritrary
Threat Model Server Malformed Inputs Integrity Checks


He et.al [41] × × × ×
FLGuard [67] × × × ×
RoFL [23] × ✓ × ×
BREA* [80] ✓ ✓ ✓ ×
EIFFeL(Our) ✓ ✓ ✓ ✓

*Has additional privacy leakage


**Secure Aggregation.** Prior work has addressed the problem of
(non-Byzantine) secure aggregation in FL [ 3, 9, 17, 81 ]. A popular
approach is to use pairwise random masking to protect the local updates [ 3, 17 ]. Advancements have been made in the communication
overhead [18, 48, 81].


**Robust Machine Learning.** A large number of studies have explored methods to make machine learners robust to Byzantine


CCS ’22, November 7–11, 2022, Los Angeles, CA, USA Roy Chowdhury et al.



failures [ 5, 10, 45 ]. Many of these robust machine-learning methods require the learned to have full access to the training data or
to fully control the training process [ 29, 39, 56, 79, 83, 88 ] which
is infeasible in FL. Another line of work has focused on the de
velopment of estimators that are inherently robust to Byzantine
errors [ 15, 25, 68, 70, 94 ]. In our work, we target a set of methods
that provides robustness by checking per-client updates [ 15, 36, 78 ].


**Verifying Data Integrity in Secure Aggregation.** Table 3 compares EIFFeL with prior work. There are three key differences between RoFL [ 23 ] and EIFFeL: ( 1 ) RoFL is designed only for range
checks with _ℓ_ 2 or _ℓ_ ∞ norms. Specifically, RoFL uses Bulletproofs
which is especially performant for range proofs (range proofs can
be aggregated where one can prove that _𝑛_ commitments lie within
a given range by providing only an additive _𝑂_ ( _𝑙𝑜𝑔_ ( _𝑛_ )) group elements over the length of a single proof). RoFL’s performance is
primarily based on this aspect of Bulletproof and all of its optimizations work only for range proofs. As such RoFL cannot support any
other checks with the same performance as currently reported in
the paper. By contrast, EIFFeL is a general framework that supports
arbitrary validation predicates with good performance. ( 2 ) RoFL
is susceptible to DoS attacks because it _only_ detects malformed
updates and aborts if it finds one. Specifically, the recovery of the
final aggregate in RoFL requires a step of nonce cancellation that
involves all the inputs by design. Hence, even if one of the input is
invalid, the final aggregate will be ill-formed. By contrast, EIFFeL is
a SAVI protocol that detects and removes malformed updates in every round. ( 3 ) RoFL assumes an honest-but-curious server, whereas
EIFFeL considers a malicious threat model. BREA [ 80 ] also removes
outlying updates but, unlike EIFFeL, it leaks pairwise distances
between inputs. Alternative solutions [ 41, 67 ] for distance-based
Byzantine-robust aggregation uses two non-colluding servers in
the semi-honest threat model, which is incompatible with FL.


**10** **Conclusion**


Practical FL settings need to ensure both the privacy and integrity
of model updates provided by the clients. In this paper, we have
formalized these goals in a new protocol, SAVI, that securely aggregates _only_ well-formed inputs ( _i.e._, updates). To demonstrate
the feasibility of SAVI, we have proposed EIFFeL: a system that
efficiently instantiates a SAVI protocol.


**References**


[[1] https://libntl.org/.](https://libntl.org/)

[2] Youtube system requirements. [https://support.google.com/youtube/answer/](https://support.google.com/youtube/answer/78358?hl=en)
[78358?hl=en.](https://support.google.com/youtube/answer/78358?hl=en)

[3] Gergely Ács and Claude Castelluccia. I have a dream! differentially private smart
metering. In _Proceedings of the 13th International Conference on Information_
_Hiding_, IH’11, page 118–132, Berlin, Heidelberg, 2011. Springer-Verlag.

[4] Nir Ailon and Bernard Chazelle. Approximate nearest neighbors and the fast
johnson-lindenstrauss transform. In _Proceedings of the Thirty-Eighth Annual_
_ACM Symposium on Theory of Computing_, STOC ’06, page 557–563, New York,
NY, USA, 2006. Association for Computing Machinery.

[5] Eugene Bagdasaryan, Andreas Veit, Yiqing Hua, Deborah Estrin, and Vitaly
Shmatikov. How to backdoor federated learning. In _arXiv:1807.00459_, 2018.

[6] Raef Bassily, Albert Cheu, Shay Moran, Aleksandar Nikolov, Jonathan Ullman,
and Zhiwei Steven Wu. Private query release assisted by public data. In _ICML_,
2020.

[7] Donald Beaver. Efficient multiparty protocols using circuit randomization. In
Joan Feigenbaum, editor, _Advances in Cryptology — CRYPTO ’91_, pages 420–432,
Berlin, Heidelberg, 1992. Springer Berlin Heidelberg.




[8] Amos Beimel, Aleksandra Korolova, Kobbi Nissim, Or Sheffet, and Uri Stemmer.
The power of synergy in differential privacy: Combining a small curator with
local randomizers. In _ITC_, 2020.

[9] James Henry Bell, Kallista A. Bonawitz, Adrià Gascón, Tancrède Lepoint, and Mariana Raykova. Secure single-server aggregation with (poly)logarithmic overhead.
In _Proceedings of the 2020 ACM SIGSAC Conference on Computer and Communica-_
_tions Security_, CCS ’20, page 1253–1269, New York, NY, USA, 2020. Association
for Computing Machinery.

[10] Arjun Nitin Bhagoji, Supriyo Chakraborty, Prateek Mittal, and Seraphin Calo.
Analyzing federated learning through an adversarial lens. In _Proceedings of the_
_International Conference on Machine Learning_, pages 634–643, 2019.

[11] Abhishek Bhowmick, John C. Duchi, Julien Freudiger, Gaurav Kapoor, and
Ryan M. Rogers. Protection against reconstruction and its applications in private
federated learning. _ArXiv_, abs/1812.00984, 2018.

[12] Battista Biggio, Blaine Nelson, and Pavel Laskov. Poisoning attacks against support vector machines. In _Proceedings of the International Coference on International_
_Conference on Machine Learning_, pages 1467–1474, 2012.

[13] Richard E. Blahut. Theory and practice of error control codes. 1983.

[14] P. Blanchard, E. M. E. Mhamdi, R. Guerraoui, and J. Stainer. Byzantine-tolerant
machine learning. In _arXiv:1703.02757_, 2017.

[15] Peva Blanchard, El Mahdi El Mhamdi, Rachid Guerraoui, and Julien Stainer.
Machine learning with adversaries: Byzantine tolerant gradient descent. In
_Advances in Neural Information Processing Systems_, pages 118–128, 2017.

[16] Keith Bonawitz, Hubert Eichner, Wolfgang Grieskamp, Dzmitry Huba, Alex
Ingerman, Vladimir Ivanov, Chloe Kiddon, Jakub Konečný, Stefano Mazzocchi,
H. Brendan McMahan, Timon Van Overveldt, David Petrou, Daniel Ramage, and
Jason Roselander. Towards federated learning at scale: System design, 2019.

[17] Keith Bonawitz, Vladimir Ivanov, Ben Kreuter, Antonio Marcedone, H. Brendan
McMahan, Sarvar Patel, Daniel Ramage, Aaron Segal, and Karn Seth. Practical
secure aggregation for privacy-preserving machine learning. In _Proceedings of_
_the ACM SIGSAC Conference on Computer and Communications Security_, pages
1175–1191, 2017.

[18] Keith Bonawitz, Fariborz Salehi, Jakub Konecný, H. Brendan McMahan, and
Marco Gruteser. Federated learning with autotuned communication-efficient
secure aggregation. _2019 53rd Asilomar Conference on Signals, Systems, and_
_Computers_, pages 1222–1226, 2019.

[19] Dan Boneh, Elette Boyle, Henry Corrigan-Gibbs, Niv Gilboa, and Yuval Ishai.
Zero-knowledge proofs on secret-shared data via fully linear pcps. In _CRYPTO_,
2019.

[20] Dan Boneh, Rosario Gennaro, Steven Goldfeder, Aayush Jain, Sam Kim, Peter
M. R. Rasmussen, and Amit Sahai. Threshold cryptosystems from threshold fully
homomorphic encryption. In _Advances in Cryptology – CRYPTO 2018: 38th Annual_
_International Cryptology Conference, Santa Barbara, CA, USA, August 19–23, 2018,_
_Proceedings, Part I_, page 565–596, Berlin, Heidelberg, 2018. Springer-Verlag.

[21] Gabriel Bracha and Sam Toueg. Asynchronous consensus and broadcast protocols.
_J. ACM_, 32(4):824–840, oct 1985.

[22] Zvika Brakerski, Craig Gentry, and Vinod Vaikuntanathan. (leveled) fully homomorphic encryption without bootstrapping. In _Proceedings of the 3rd Innovations_
_in Theoretical Computer Science Conference_, ITCS ’12, page 309–325, New York,
NY, USA, 2012. Association for Computing Machinery.

[23] Lukas Burkhalter, Hidde Lycklama, Alexander Viand, Nicolas Küchler, and Anwar Hithnawi. Rofl: Attestable robustness for secure federated learning. In
_arXiv:2107.03311_, 2021.

[24] Xiaoyu Cao, Minghong Fang, Jia Liu, and Neil Zhenqiang Gong. Fltrust:
Byzantine-robust federated learning via trust bootstrapping. 2021.

[25] Lingjiao Chen, Hongyi Wang, Zachary Charles, and Dimitris Papailiopoulos.
Draco: Byzantine-resilient distributed training via redundant gradients. In _Pro-_
_ceedings of the International Conference on Machine Learning_, 2018.

[26] Xinyun Chen, Chang Liu, Bo Li, Kimberly Lu, and Dawn Song. Targeted backdoor
attacks on deep learning systems using data poisoning. In _arXiv:1712.05526_, 2017.

[27] Gregory Cohen, Saeed Afshar, Jonathan Tapson, and André van Schaik. Emnist:
Extending mnist to handwritten letters. In _2017 International Joint Conference on_
_Neural Networks (IJCNN)_, pages 2921–2926, 2017.

[28] Henry Corrigan-Gibbs and Dan Boneh. Prio: Private, robust, and scalable computation of aggregate statistics. In _Proceedings of the USENIX Symposium on_
_Networked Systems Design and Implementation_, 2017.

[29] Gabriela F. Cretu, Angelos Stavrou, Michael E. Locasto, Salvatore J. Stolfo, and
Angelos D. Keromytis. Casting out demons: Sanitizing training data for anomaly
sensors. In _IEEE Symposium on Security and Privacy (SP)_, pages 81–95, 2008.

[30] Scott A. Crosby and Dan S. Wallach. Efficient data structures for tamper-evident
logging. In _Proceedings of the 18th Conference on USENIX Security Symposium_,
SSYM’09, page 317–334, USA, 2009. USENIX Association.

[31] Georgios Damaskinos, El Mahdi El Mhamdi, Rachid Guerraoui, Rhicheek Patra,
and Mahsa Taziki. Asynchronous byzantine machine learning (the case of sgd).
In _ICML_, 2018.

[32] W. Diffie and M. Hellman. New directions in cryptography. _IEEE Transactions on_
_Information Theory_, 22(6):644–654, 1976.


E FFeL: Ensuring Integrity For Federated Learning CCS ’22, November 7–11, 2022, Los Angeles, CA, USA




[33] El Mahdi El Mhamdi, Rachid Guerraoui, and Sébastien Rouault. The hidden
vulnerability of distributed learning in Byzantium. In Jennifer Dy and Andreas
Krause, editors, _Proceedings of the 35th International Conference on Machine Learn-_
_ing_, volume 80 of _Proceedings of Machine Learning Research_, pages 3521–3530.
PMLR, 10–15 Jul 2018.

[34] Minghong Fang, Xiaoyu Cao, Jinyuan Jia, and Neil Zhenqiang Gong. Local model
poisoning attacks to byzantine-robust federated learning. In _USENIX Security_
_Symposium_, 2020.

[35] Paul Feldman. A practical scheme for non-interactive verifiable secret sharing.
In _28th Annual Symposium on Foundations of Computer Science (sfcs 1987)_, pages
427–438, 1987.

[36] Clement Fung, Chris J.M. Yoon, and Ivan Beschastnikh. Mitigating sybils in
federated learning poisoning. In _arXiv:1808.04866_, 2018.

[37] Shuhong Gao. _A New Algorithm for Decoding Reed-Solomon Codes_, pages 55–68.
Springer US, Boston, MA, 2003.

[38] Joachim von zur Gathen and Jrgen Gerhard. _Modern Computer Algebra_ . Cambridge University Press, USA, 3rd edition, 2013.

[39] Tianyu Gu, Brendan Dolan-Gavitt, and Siddharth Garg. Badnets: Identifying
vulnerabilities in the machine learning model supply chain. In _arXiv:1708.06733_,
2017.

[40] Kaiming He, X. Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning
for image recognition. _2016 IEEE Conference on Computer Vision and Pattern_
_Recognition (CVPR)_, pages 770–778, 2016.

[41] Lie He, Sai Praneeth Karimireddy, and Martin Jaggi. Secure byzantine-robust
machine learning, 2020.

[42] Aayush Jain, Peter M. R. Rasmussen, and Amit Sahai. Threshold fully homomorphic encryption. Cryptology ePrint Archive, Report 2017/257, 2017.
[https://ia.cr/2017/257.](https://ia.cr/2017/257)

[43] Jinyuan Jia, Xiaoyu Cao, and Neil Zhenqiang Gong. Intrinsic certified robustness
of bagging against data poisoning attacks. In _AAAI_, 2021.

[44] Peter Kairouz, Ziyu Liu, and Thomas Steinke. The distributed discrete gaussian
mechanism for federated learning with secure aggregation. _ArXiv_, abs/2102.06387,
2021.

[45] Peter Kairouz, H. Brendan McMahan, Brendan Avent, Aurelien Bellet, Mehdi Bennis, Arjun Nitin Bhagoji, Kallista Bonawitz, Zachary Charles, Graham Cormode,
Rachel Cummings, Rafael G.L. D’Oliveira, Hubert Eichner, Salim El Rouayheb,
David Evans, Josh Gardner, Zachary Garrett, Adria Gascon, Badih Ghazi, Phillip B.
Gibbons, Marco Gruteser, Zaid Harchaoui, Chaoyang He, Lie He, Zhouyuan Huo,
Ben Hutchinson, Justin Hsu, Martin Jaggi, Tara Javidi, Gauri Joshi, Mikhail Khodak, Jakub Konecny, Aleksandra Korolova, Farinaz Koushanfar, Sanmi Koyejo,
Tancrede Lepoint, Yang Liu, Prateek Mittal, Mehryar Mohri, Richard Nock, Ayfer
Ozgur, Rasmus Pagh, Hang Qi, Daniel Ramage, Ramesh Raskar, Mariana Raykova,
Dawn Song, Weikang Song, Sebastian U. Stich, Ziteng Sun, Ananda Theertha
Suresh, Florian Tramer, Praneeth Vepakomma, Jianyu Wang, Li Xiong, Zheng Xu,
Qiang Yang, Felix X. Yu, Han Yu, and Sen Zhao. Advances and open problems in
federated learning. In _arXiv:1912.04977_, 2019.

[46] Jonathan Katz and Yehuda Lindell. _Introduction to Modern Cryptography, Second_
_Edition_ . Chapman & Hall/CRC, 2nd edition, 2014.

[47] Jakub Konečný, H. Brendan McMahan, Felix X. Yu, Peter Richtárik,
Ananda Theertha Suresh, and Dave Bacon. Federated learning: Strategies for
improving communication efficiency. _CoRR_, abs/1610.05492, 2016.

[48] Jakub Konecný, H. Brendan McMahan, Felix X. Yu, Peter Richtárik,
Ananda Theertha Suresh, and Dave Bacon. Federated learning: Strategies for
improving communication efficiency. _ArXiv_, abs/1610.05492, 2016.

[49] Alex Krizhevsky. The cifar-10 dataset.

[50] Y. Lecun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied
to document recognition. _Proceedings of the IEEE_, 86(11):2278–2324, 1998.

[51] Yann LeCun, Corinna Cortes, and Christopher J.C. Burges. The mnist database
of handwritten digits.

[52] Alexander Levine and Soheil Feizi. Provable adversarial robustness for fractional
lp threat models. In Gustau Camps-Valls, Francisco J. R. Ruiz, and Isabel Valera,
editors, _International Conference on Artificial Intelligence and Statistics, AISTATS_
_2022, 28-30 March 2022, Virtual Event_, volume 151 of _Proceedings of Machine_
_Learning Research_, pages 9908–9942. PMLR, 2022.

[53] Liping Li, Wei Xu, Tianyi Chen, Georgios Giannakis, and Qing Ling. Rsa:
Byzantine-robust stochastic aggregation methods for distributed learning from
heterogeneous datasets. _Proceedings of the AAAI Conference on Artificial Intelli-_
_gence_, 33:1544–1551, 07 2019.

[54] Suyi Li, Yong Cheng, Wei Wang, Yang Liu, and Tianjian Chen. Learning to detect
malicious clients for robust federated learning. _CoRR_, abs/2002.00211, 2020.

[55] Shu Lin and Daniel J. Costello. _Error control coding: fundamentals and applications_ .
Pearson/Prentice Hall, Upper Saddle River, NJ, 2004.

[56] Kang Liu, Brendan Dolan-Gavitt, and Siddharth Garg. Fine-pruning: Defending
against backdooring attacks on deep neural networks. pages 273–294, 2018.

[57] Terrance Liu, Giuseppe Vietri, Thomas Steinke, Jonathan Ullman, and Zhiwei Steven Wu. Leveraging public data for practical private query release, 2021.

[58] [Wolfram Mathworld. Lagrange interpolating polynomial. https://mathworld.](https://mathworld.wolfram.com/LagrangeInterpolatingPolynomial.html)
[wolfram.com/LagrangeInterpolatingPolynomial.html.](https://mathworld.wolfram.com/LagrangeInterpolatingPolynomial.html)




[59] R. J. McEliece. The guruswami–sudan decoding algorithm for reed–solomon
codes, 2003.

[60] Brendan McMahan and Daniel Ramage. Federated learning: Collaborative machine learning without centralized training data, 2017.

[61] H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and
Blaise Agüera y Arcas. Communication-efficient learning of deep networks
from decentralized data. In _Proceedings of the International Conference on Artifi-_
_cial Intelligence and Statistics_, 2017.

[62] Shike Mei and Xiaojin Zhu. Using machine teaching to identify optimal trainingset attacks on machine learners. In _Proceedings of the AAAI Conference on Artificial_
_Intelligence_, pages 2871–2877, 2015.

[63] Luca Melis, Congzheng Song, Emiliano De Cristofaro, and Vitaly Shmatikov.
Exploiting unintended feature leakage in collaborative learning. In _2019 IEEE_
_Symposium on Security and Privacy (SP)_, pages 691–706, 2019.

[64] Mohammad Naseri, Jamie Hayes, and Emiliano De Cristofaro. Local and central
differential privacy for robustness and privacy in federated learning, 2021.

[65] Milad Nasr, Reza Shokri, and Amir Houmansadr. Comprehensive privacy analysis of deep learning: Passive and active white-box inference attacks against
centralized and federated learning. _2019 IEEE Symposium on Security and Privacy_
_(SP)_, May 2019.

[66] Jelani Nelson. Sketching algorithms.

[67] Thien Duc Nguyen, Phillip Rieger, Hossein Yalame, Helen Möllering, Hossein
Fereidooni, Samuel Marchal, Markus Miettinen, Azalia Mirhoseini, Ahmad-Reza
Sadeghi, Thomas Schneider, and Shaza Zeitouni. Flguard: Secure and private
federated learning, 2021.

[68] Xudong Pan, Mi Zhang, Duocai Wu, Qifan Xiao, Shouling Ji, and Zhemin Yang.
Justinian’s GAAvernor: Robust distributed learning with gradient aggregation
agent. In _USENIX Security_, pages 1641–1658, 2020.

[69] Ashwinee Panda, Saeed Mahloujifar, Arjun Nitin Bhagoji, Supriyo Chakraborty,
and Prateek Mittal. Sparsefed: Mitigating model poisoning attacks in federated
learning with sparsification. In Gustau Camps-Valls, Francisco J. R. Ruiz, and
Isabel Valera, editors, _Proceedings of The 25th International Conference on Artificial_
_Intelligence and Statistics_, volume 151 of _Proceedings of Machine Learning Research_,
pages 7587–7624. PMLR, 28–30 Mar 2022.

[70] Shashank Rajput, Hongyi Wang, Zachary Charles, and Dimitris Papailiopoulos.
Detox: A redundancy-based framework for faster and more robust gradient
aggregation. 2019.

[71] Elan Rosenfeld, Ezra Winston, Pradeep Ravikumar, and J. Zico Kolter. Certified
robustness to label-flipping attacks via randomized smoothing. In _ICML_, 2020.

[72] Edo Roth, Daniel Noble, Brett Hemenway Falk, and Andreas Haeberlen. Honeycrisp: Large-scale differentially private aggregation without a trusted core. In
_Proceedings of the 27th ACM Symposium on Operating Systems Principles_, SOSP ’19,
page 196–210, New York, NY, USA, 2019. Association for Computing Machinery.

[73] Berry Schoenmakers. A simple publicly verifiable secret sharing scheme and its
application to electronic voting. In _In CRYPTO_, pages 148–164. Springer-Verlag,
1999.

[74] J. T. Schwartz. Fast probabilistic algorithms for verification of polynomial identities. _J. ACM_, 27(4):701–717, October 1980.

[75] Adi Shamir. How to share a secret. _Commun. ACM_, 22(11):612–613, November

1979.

[76] Virat Shejwalkar and Amir Houmansadr. Manipulating the byzantine: Optimizing
model poisoning attacks and defenses for federated learning. In _NDSS_, 2021.

[77] Virat Shejwalkar and Amir Houmansadr. Manipulating the byzantine: Optimizing
model poisoning attacks and defenses for federated learning. In _NDSS_, 2021.

[78] Shiqi Shen, Shruti Tople, and Prateek Saxena. Auror: Defending against poisoning
attacks in collaborative deep learning systems. In _ACM ACSAC_, pages 508–519,
2016.

[79] Yanyao Shen and Sujay Sanghavi. Learning with bad training data via iterative
trimmed loss minimization. In _International Conference on Machine Learning_
_(ICML)_, pages 5739–5748, 2019.

[80] Jinhyun So, Basak Guler, and A. Salman Avestimehr. Byzantine-resilient secure
federated learning. _IEEE Journal in Selected Areas in Communications: Machine_
_Learning in Communications and Networks_, 2020.

[81] Jinhyun So, Basak Guler, and A. Salman Avestimehr. Turbo-aggregate: Breaking
the quadratic aggregation barrier in secure federated learning, 2021.

[82] Markus Stadler. Publicly verifiable secret sharing. pages 190–199. Springer-Verlag,
1996.

[83] Jacob Steinhardt, Pang Wei W. Koh, and Percy S. Liang. Certified defenses for
data poisoning attacks. In _Advances in Neural Information Processing Systems_
_(NeurIPS)_, pages 3517–3529, 2017.

[84] Ziteng Sun, Peter Kairouz, Ananda Theertha Suresh, and H. Brendan McMahan.
Can you really backdoor federated learning? In _arXiv:1911.07963_, 2019.

[85] Ziteng Sun, Peter Kairouz, Ananda Theertha Suresh, and H. Brendan McMahan.
Can you really backdoor federated learning? _ArXiv_, abs/1911.07963, 2019.

[86] Chunming Tang, Dingyi Pei, and Zhuojun Liu Yong He. Non-interactive and
information-theoretic secure publicly verifiable secret sharing.

[87] Binghui Wang, Xiaoyu Cao, Jinyuan jia, and Neil Zhenqiang Gong. On certifying
robustness against backdoor attacks via randomized smoothing, 2020.


CCS ’22, November 7–11, 2022, Los Angeles, CA, USA Roy Chowdhury et al.


[88] Bolun Wang, Yuanshun Yao, Shawn Shan, Huiying Li, Bimal Viswanath, Haitao
Zheng, and Ben Y. Zhao. Neural cleanse: Identifying and mitigating backdoor
attacks in neural networks. In _IEEE Symposium on Security and Privacy (SP)_,
pages 707–723, 2019.

[89] Chulin Xie, Minghao Chen, Pin-Yu Chen, and Bo Li. Crfl: Certifiably robust
federated learning against backdoor attacks. In Marina Meila and Tong Zhang,
editors, _Proceedings of the 38th International Conference on Machine Learning_,
volume 139 of _Proceedings of Machine Learning Research_, pages 11372–11382.
PMLR, 18–24 Jul 2021.

[90] Chulin Xie, Keli Huang, Pin-Yu Chen, and Bo Li. Dba: Distributed backdoor
attacks against federated learning. In _ICLR_, 2020.

[91] Cong Xie. Zeno++: robust asynchronous SGD with arbitrary number of byzantine
workers. _CoRR_, abs/1903.07020, 2019.

[92] Cong Xie, Oluwasanmi Koyejo, and Indranil Gupta. Zeno: Distributed stochastic
gradient descent with suspicion-based fault-tolerance. In _Proceedings of the_
_International Conference on Machine Learning_, 2019.

[93] Cong Xie, Oluwasanmi Koyejo, and Indranil Gupta. Zeno++: Robust fully asynchronous SGD. In _Proceedings of the International Conference on Machine Learning_,
2020.

[94] Dong Yin, Yudong Chen, Ramchandran Kannan, and Peter Bartlett. Byzantinerobust distributed learning: Towards optimal statistical rates. In _International_
_Conference on Machine Learning (ICML)_, 2019.

[95] Hongxu Yin, Arun Mallya, Arash Vahdat, José Manuel Álvarez, Jan Kautz, and
Pavlo Molchanov. See through gradients: Image batch recovery via gradinversion.
_2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_,
pages 16332–16341, 2021.

[96] Zalando. Fashion mnist.

[97] Ligeng Zhu, Zhijian Liu, and Song Han. Deep leakage from gradients. In _NeurIPS_,
2019.

[98] Richard Zippel. Probabilistic algorithms for sparse polynomials. In _Proceed-_
_ings of the International Symposiumon on Symbolic and Algebraic Computation_,
EUROSAM ’79, page 216–226, Berlin, Heidelberg, 1979. Springer-Verlag.


E FFeL: Ensuring Integrity For Federated Learning CCS ’22, November 7–11, 2022, Los Angeles, CA, USA



**Table 4: Notations**


Symbol Description


_𝑛_ Total number of clients

_𝑚_ Number of malicious clients

S Server

C _𝑖_ _𝑖_ -th client
_𝐷_ _𝑖_ Private dataset of C _𝑖_

C Set of all _𝑛_ clients

C _𝐻_ Set of _𝑛_ − _𝑚_ honest clients
C _𝑀_ Set of _𝑚_ malicious clients
Valid(·) Validation predicate
M Global model to be trained

_𝑢_ _𝑖_ Local update (gradient) of client C _𝑖_
U Aggregate update
C Valid Set of clients such that for all C _𝑖_ ∈C Valid, Valid( _𝑢_ _𝑖_ ) = 1
U Valid Aggregate of valid updates only U Valid = [�] C _𝑖_ ∈C Valid _[𝑢]_ _𝑖_
_𝜅_ Security parameter
( _𝑖,𝑠_ _𝑖_ ) _𝑖_ -th Shamir’s secret share for a secret _𝑠_ ∈ F
Ψ Check string for the verifiable secret sharing
_𝑝𝑝_ Public parameters of the cryptographic protocols
_𝑝𝑘_ Public key
_𝑠𝑘_ Secret key
_𝑠𝑘_ _𝑖𝑗_ Shared secret key between clients C _𝑖_ and C _𝑗_
P Prover in the SNIP protocol
V _𝑖_ _𝑖_ -th verifier in the SNIP protocol
_𝜋_ SNIP proof
_ℎ_ / _𝑓_ / _𝑔_ Polynomials generated by P for the construction of _𝜋_
( _𝑎,𝑏,𝑐_ ) Beaver’s triplet generated by P for the construction of _𝜋_

[ _𝑠_ ] _𝑖_ _𝑖_ -th additive secret share for a secret _𝑠_ ∈ F
_𝑤_ _[𝑜𝑢𝑡]_ Value of the output wire of the circuit Valid(·)
_𝜎_ Proof summary broadcasted by the verifiers
F A prime field
M Number of multiplication gates in Valid(·)
B Public bulletin board
C \ _𝑖_ Set of all clients except C _𝑖_, C \ _𝑖_ = C \ C _𝑖_
C [∗] List of malicious clients maintained by S in EIFFeL
( _𝑗,𝑠_ _𝑖𝑗_ ) Client C _𝑗_ ’s (Shamir secret) share of client C _𝑖_ ’s secret _𝑠_ _𝑖_ ∈ F in EIFFeL
_𝜋_ _𝑖_ Client C _𝑖_ ’s proof in EIFFeL
_ℎ_ _𝑖_ / _𝑓_ _𝑖_ / _𝑔_ _𝑖_ Polynomials generated by client C _𝑖_ for the construction of _𝜋_ _𝑖_ in EIFFeL
( _𝑎_ _𝑖_ _,𝑏_ _𝑖_ _,𝑐_ _𝑖_ ) Beaver’s triplet generated by client C _𝑖_ for the construction of _𝜋_ _𝑖_ in EIFFeL
_𝑤_ _𝑖_ _[𝑜𝑢𝑡]_ Value of the output wire of the circuit Valid( _𝑢_ _𝑖_ ) for client C _𝑖_
Ψ _𝜋_ _𝑖_ Check string generated by client C _𝑖_ for the shares of their proof _𝜋_ _𝑖_
Ψ _𝑢_ _𝑖_ Check string generated by client C _𝑖_ for the shares of their update _𝑢_ _𝑖_
_𝜎_ _𝑗𝑖_ Client C _𝑗_ ’s shar of the summary for client C _𝑖_ ’s proof in EIFFeL
_𝜆_ _𝑗𝑖_ Client C _𝑗_ ’s share of the random digest for client C _𝑖_ ’s proof in EIFFeL
_𝜆_ _𝑖_ Client C _𝑖_ ’s random digest reconstructed from the shares { _𝜆_ _𝑗𝑖_ } _, 𝑗_ ∈C \ _𝑖_
_𝜎_ _𝑖_ Client C _𝑖_ ’s proof summary reconstructed from the shares { _𝜎_ _𝑗𝑖_ } _, 𝑗_ ∈C \ _𝑖_

**11** **Appendix**


**11.1** **Building Blocks Cntd.**


**Arithmetic Circuit.** An arithmetic circuit, C : F _[𝑘]_ ↦→ F, represents a
computation over a finite field F . It can be represented by a directed
acyclic graph (DAG) consisting of three types of nodes: ( 1 ) inputs,
( 2 ) gates and ( 3 ) outputs. Input nodes have in-degree zero and outdegree one: the _𝑘_ input nodes return input variables { _𝑥_ 1 _,_ - · · _,𝑥_ _𝑘_ }
with _𝑥_ _𝑖_ ∈ F . Gate nodes have in-degree two and out-degree one;
they perform either the + operation (addition gate) or the × operation (multiplication gate). Every circuit has a single output node
with out-degree zero. A circuit is evaluated by traversing the DAG,
starting from the inputs, and assigning a value in F to every wire
until the output node is evaluated.


**Shamir’s Secret Sharing Scheme.** The scheme is _additive_, _i.e._, it
allows addition of two secret shared values locally. Formally, for all
_𝑠,𝑤_ ∈ F and _𝑄_ ⊆ _𝑃,_ | _𝑄_ | ≤ _𝑡_ :


_𝑠_ + _𝑤_ ← SS.recon({( _𝑖,𝑠_ _𝑖_ + _𝑤_ _𝑖_ ) _𝑖_ ∈ _𝑄_ }) (7)


Additionally, the scheme is a linear secret sharing scheme which
means that any linear operations performed on the individual shares



translates to operations performed on the secret, upon reconstruction. Specifically, for _𝑄_ ⊆ _𝑃,_ | _𝑄_ | ≥ _𝑡_ and _𝛼, 𝛽_ ∈ F:


_𝛼𝑠_ + _𝛽_ ← SS.recon({( _𝑖, 𝛼𝑠_ _𝑖_ + _𝛽_ ) _𝑖_ ∈ _𝑄_ }) (8)


This means that a party can perform linear operations on the secret
_locally_ .


_Verifiable Secret Shares._ To make the Shamir’s Secret shares verifiable, we use Feldman’s [ 35 ] VSS technique. Let _𝑐_ ( _𝑥_ ) = _𝑐_ 0 + _𝑐_ 1 _𝑥_ +

- · · _𝑐_ _𝑡_ −1 _𝑥_ _[𝑡]_ [−][1] denote the polynomial used in generating the shares
where _𝑐_ 0 = _𝑠_ is the secret. The check string are the commitments
to the coeffcients given by


_𝜓_ _𝑖_ = _𝑔_ _[𝑐]_ _[𝑖]_ _,𝑖_ ∈{0 _,_        - · · _,𝑡_ − 1} (9)


where _𝑔_ denotes a generator of F . All arithmetic is taken modulo _𝑞_
such that ( _𝑝_ | _𝑞_ − 1) where _𝑝_ is the prime of F.


For verifiying a share ( _𝑗,𝑠_ _𝑗_ ), a party needs to check whether _𝑔_ _[𝑠]_ _[𝑗]_ =
� _𝑖𝑡_ =−01 _[𝜓]_ _𝑖_ _[𝑗]_ _[𝑖]_ [. The privacy of the secret] _[ 𝑠]_ [=] _[ 𝑐]_ [0] [ is implied by the the]
intractability of computing discrete logarithms [35].


**Short Non-Interactive Proofs (SNIP).** Here, we detail the SNIP
protocols. SNIP works in two stages as follows:


_(1) Generation of Proof._ The prover P evaluates the circuit Valid(·)
on its input _𝑥_ to obtain the value that every wire in the circuit takes
on during the computation of Valid( _𝑥_ ) . Using these wire values,
P constructs three randomized polynomials _𝑓_, _𝑔_, and _ℎ_, which
encode the values of the input and output wires of each of the M
multiplication gates in the computation of Valid( _𝑥_ ).


Let us label the _𝑀_ multiplication gates in the Valid(·) circuit in
the topological order from inputs to outputs as { 1 _,_ - · · _,_ M} . Let
_𝑢_ _𝑡_ and _𝑣_ _𝑡_ denote the values on the left and right input wires of
the _𝑡_ -th multiplication gate for _𝑡_ ∈[ _𝑀_ ] . The prover P samples
two values _𝑢_ 0 and _𝑣_ 0 uniformly at random from F . _𝑓_ and _𝑔_ are
defined to be the lowest degree polynomials such that _𝑓_ ( _𝑡_ ) = _𝑢_ _𝑡_
and _𝑔_ ( _𝑡_ ) = _𝑣_ _𝑡_ _,_ ∀ _𝑡_ ∈[M] . Next, _ℎ_ is defined as the polynomial
_ℎ_ = _𝑓_ - _𝑔_ . The polynomials _𝑓_ and _𝑔_ has degree at most _𝑀_ and the
polynomial _ℎ_ has degree at most 2 M . It is easy to see that _ℎ_ ( _𝑡_ )
is the value of the output wire ( _𝑢_ _𝑡_ - _𝑣_ _𝑡_ ) of the _𝑡_ -th multiplication
gate in the Valid( _𝑥_ ) circuit since _ℎ_ ( _𝑡_ ) = _𝑓_ ( _𝑡_ ) · _𝑔_ ( _𝑡_ ) = _𝑢_ _𝑡_ - _𝑣_ _𝑡_ _,_ ∀ _𝑡_ ∈

[M] . The prover P can construct the polynomials _𝑓_ and _𝑔_ using
polynomial interpolation and can multiply them produce _ℎ_ = _𝑓_ _𝑔_ . Additionally, P samples a single set of Beaver’s multiplication
triples [ 7 ]: ( _𝑎,𝑏,𝑐_ ) ∈ F [3] such that _𝑎_ - _𝑏_ = _𝑐_ ∈ F . Prover P constructs
the proof share [ _𝜋_ ] _𝑖_ = ⟨[ _𝑓_ ( 0 )] _𝑖_ _,_ [ _𝑔_ ( 0 )] _𝑖_ _,_ [ _ℎ_ ] _𝑖_ _,_ ([ _𝑎_ ] _𝑖_ _,_ [ _𝑏_ ] _𝑖_ _,_ [ _𝑐_ ] _𝑖_ )⟩ [1] for
verifier V _𝑖_ by splitting:


 - the random values _𝑓_ ( 0 ) = _𝑢_ 0 and _𝑔_ ( 0 ) = _𝑣_ 0, using additive
secret sharing,


 - the coefficients of _ℎ_ (denoted by [ _ℎ_ ] _𝑖_, and


 - the sampled Beaver’s triplets ( _𝑎,𝑏,𝑐_ ).


The prover then sends the respective shares of the input and the
proof ([ _𝑥_ ] _𝑖_ _,_ [ _𝜋_ ] _𝑖_ ) to each of the verifiers V _𝑖_ .


1 Note that we omitted the terms [ _𝑓_ (0)] _𝑖_ and [ _𝑔_ (0)] _𝑖_ from _𝜋_ _𝑖_ in Sec. 4.1 for the ease
of exposition.


CCS ’22, November 7–11, 2022, Los Angeles, CA, USA Roy Chowdhury et al.



_(2) Verification of Proof._ Using [ _𝑥_ ] _𝑖_, the share of the provers’s
private value _𝑥_, and [ _𝑓_ ( 0 )] _𝑖_, [ _𝑔_ ( 0 )] _𝑖_, and [ _ℎ_ ] _𝑖_, each verifier V _𝑖_ can
_locally_ (i.e., without communicating with the other verifiers/prover)
produce shares [ _𝑓_ ] _𝑖_ and [ _𝑔_ ] _𝑖_ of the polynomials _𝑓_ and _𝑔_ as follows:


 - V _𝑖_ reconstructs a share of every wire for the Valid( _𝑥_ ) circuit.
This is possible since V _𝑖_ has access to ( 1 ) a share of each of
the input wire values ([ _𝑥_ ] _𝑖_ ) and ( 2 ) a share of each wire value
coming out of a multiplication gate ([ _ℎ_ ] _𝑖_ ( _𝑡_ ) _,𝑡_ ∈[M] is a share
of the _𝑡_ -th such wire). Hence, V _𝑖_ can derive all other wire value
shares via affine operations on the wire value shares it already
has.


 - Using these wire value shares and shares of _𝑓_ (0) and _𝑔_ (0), V _𝑖_
uses polynomial interpolation to construct [ _𝑓_ ] _𝑖_ and [ _𝑔_ ] _𝑖_


To verify that Valid( _𝑥_ ) = 1 and hence, accept the input _𝑥_, the
verifiers need to check two things:


 - check the consistency of P’s computation of Valid( _𝑥_ ), and


 - check that the value of final output wire of the computation,
Valid( _𝑥_ ), denoted by _𝑤_ _[𝑜𝑢𝑡]_ is indeed 1.


For carrying out the above mentioned checks, the verifier V _𝑖_ broadcasts a summary _𝜎_ _𝑖_ = ([ _𝑤_ _[𝑜𝑢𝑡]_ ] _𝑖_ _,_ [ _𝜆_ ] _𝑖_ ), where [ _𝑤_ _[𝑜𝑢𝑡]_ ] _𝑖_ is V _𝑖_ ’s share
of the output wire and [ _𝜆_ ] _𝑖_ is a share of a random digest that the
verifier computes from the shares of the other wire values and the
proof share _𝜋_ _𝑖_ . The details are discussed as follows:


_(2a) Checking the Consistency of_ P _’s Computation of_ _Valid_ ( _𝑥_ ) _._ For
honest provers and verifiers, the verifiers will now hold shares of
polynomials _𝑓_, _𝑔_, and _ℎ_ such that _𝑓_ - _𝑔_ = _ℎ_ . In contrast, a malicious
_ℎ_ prover could have sent the verifiers shares of a different polynomial ˆ such that, for some _𝑡_ ∈[ _𝑀_ ] _,ℎ_ ˆ ( _𝑡_ ) is not the value on the output
wire in the _𝑡_ -th multiplication gate of the V( _𝑥_ ) circuit. In this case,
the verifiers end up reconstructing shares of polynomials _𝑓_ [ˆ] and
_𝑔_ ˆ that might not be equal to _𝑓_ and _𝑔_ . Then, we have _ℎ_ [ˆ] ≠ _𝑓_ [ˆ] - ˆ _𝑔_ as
explained below. Consider the least _𝑡_ [′] for which _ℎ_ [ˆ] ( _𝑡_ [′] ) ≠ _ℎ_ ( _𝑡_ [′] ) . For
all _𝑡_ ≤ _𝑡_ [′], _𝑓_ [ˆ] ( _𝑡_ ) = _𝑓_ ( _𝑡_ ) and _𝑔_ ( _𝑡_ ) = _𝑔_ ( _𝑡_ ), by construction. Since,


ˆ
_ℎ_ ( _𝑡_ [′] ) ≠ _ℎ_ ( _𝑡_ [′] ) = _𝑓_ ( _𝑡_ [′] ) · _𝑔_ ( _𝑡_ [′] ) = ˆ _𝑓_ ( _𝑡_ ) [′]     - ˆ _𝑔_ ( _𝑡_ [′] ) _,_ (10)


it must be that _ℎ_ [ˆ] ( _𝑡_ [′] ) ≠ _𝑓_ [ˆ] ( _𝑡_ [′] ) · ˆ _𝑔_ ( _𝑡_ [′] ), so _ℎ_ [ˆ] ≠ _𝑓_ [ˆ] - ˆ _𝑔._ The verifiers can
employ the above check using the Schwartz-Zippel randomized
polynomial identity test [74, 98] as explained later in this section.


_(2b) Output Verification._ In case all the verifiers are honest, each
V _𝑖_ now holds a set of shares of the values of all the wires of the
Valid( _𝑥_ ) circuit. So to confirm that Valid( _𝑥_ ) = 1, the verifiers need
only broadcast their shares of the output wire _𝑤_ _𝑖_ _[𝑜𝑢𝑡]_ . The verifiers
can thus reconstruct its exact value from all the broadcasted shares
_𝑤_ _[𝑜𝑢𝑡]_ = [�] _[𝑘]_ _𝑖_ =1 [[] _[𝑤]_ _[𝑜𝑢𝑡]_ []] _[𝑖]_ [and check whether] _[ 𝑤]_ _[𝑜𝑢𝑡]_ [=] [ 1, in which case it]
must be that Valid( _𝑥_ ) = 1 (except with some small failure probability due to the polynomial identity test).


_Polynomial Identity Test_ . Recall that each verifier V _𝑖_ holds shares

[ _𝑓_ [ˆ] ] _𝑖_, [ _𝑔_ ˆ] _𝑖_ and [ _ℎ_ [ˆ] ] _𝑖_ of the polynomials _𝑓_ [ˆ], ˆ _𝑔_ and _ℎ_ [ˆ] . Furthermore, it
holds that _𝑓_ [ˆ] - ˆ _𝑔_ = _ℎ_ [ˆ] if and only the set of the wire value shares,
held by the verifiers, sum up to the internal wire values of the



Valid( _𝑥_ ) circuit computation. The verifiers now execute a variant
of the Schwartz-Zippel randomized polynomial identity test to
check whether this relation holds. The main idea of the test is that
if _𝑓_ [ˆ] ( _𝑡_ )· _𝑔_ ˆ( _𝑡_ ) ≠ _ℎ_ [ˆ] ( _𝑡_ ), then the polynomial _𝑡_ - ( _𝑓_ [ˆ] ( _𝑡_ )· _𝑔_ ˆ( _𝑡_ )− _ℎ_ [ˆ] ( _𝑡_ )) is a nonzero polynomial of degree at most 2 M+ 1. (The utility of multiplying
the polynomial _𝑓_ [ˆ] - ˆ _𝑔_ − _ℎ_ [ˆ] by _𝑡_ is explained in the next paragraph)
Such a polynomial can have at most 2 M + 1 zeros in F, so for a _𝑟_ ∈ F
chosen at random and after evaluating _𝑟_ - ( _𝑓_ [ˆ] ( _𝑟_ ) · ˆ _𝑔_ ( _𝑟_ ) − _ℎ_ [ˆ] ( _𝑟_ )), the
verifiers will detect that _𝑓_ [ˆ] - ˆ _𝑔_ ≠ _ℎ_ [ˆ] with probability at least 1 [2][M] |F [+] | [1] [.]


For the polynomial identity test, one of the verifiers samples a
random value _𝑟_ ∈ F and broadcasts it. Each verifier V _𝑖_ can locally
compute the shares [ _𝑓_ [ˆ] ( _𝑟_ )] _𝑖_, [ _𝑔_ ˆ( _𝑟_ )] _𝑖_, and [ _ℎ_ [ˆ] ( _𝑟_ )] _𝑖_ since polynomial
evaluation requires only affine operations. V _𝑖_ then applies a local
linear operation to these last two shares to obtain the shares [ _𝑟_ _𝑔_ ˆ( _𝑟_ )] _𝑖_ and [ _𝑟_ - _ℎ_ [ˆ] ( _𝑟_ )] _𝑖_ .


_Multiplication of Shares._ Note that the verifiers need to securely multiply their shares [ _𝑓_ [ˆ] ( _𝑟_ )] _𝑖_ and [ _𝑟_ - ˆ _𝑔_ ( _𝑟_ )] _𝑖_ to get a share [ _𝑟_ - _𝑓_ [ˆ] ( _𝑟_ )· ˆ _𝑔_ ( _𝑟_ )] _𝑖_
without leaking anything to each other about the values _𝑓_ [ˆ] ( _𝑟_ ) and
_𝑔_ ˆ( _𝑟_ ) . This can be performed via the Beaver’s MPC multiplication
protocol (described later). Using this protocol, verifiers with access to one-time-use shares ([ _𝑎_ ] _𝑖_ _,_ [ _𝑏_ ] _𝑖_ _,_ [ _𝑐_ ] _𝑖_ ) ∈ F [3] of random values
such that _𝑎_ - _𝑏_ = _𝑐_ ∈ F (“multiplication triples”), can execute a
multi-party multiplication of a pair of secret-shared values. For
SNIPs, the prover P generates the multiplication triple on behalf
of the verifiers and sends shares of these values to each verifier. If

P produces the shares of these values correctly, then the verifiers
can perform a multi-party multiplication of shares to complete
the correctness check as discussed above. More importantly, we
can ensure that even if P sends shares of an invalid multiplication
triple, the verifiers will still catch the cheating prover with high
probability. Let’s assume that the cheating prover sends the shares
([ _𝑎_ ] _𝑖_ _,_ [ _𝑏_ ] _,_ [ _𝑐_ ] _𝑖_ ) ∈ F [3] such that _𝑎_ - _𝑏_ ≠ _𝑐_ ∈ F . Let _𝑎_ - _𝑏_ = ( _𝑐_ + _𝛼_ ) ∈ F,
for some constant _𝛼_ _>_ 0. Executing the polynomial identity test
using the above triples will shift the result of the test by _𝛼_ . So the
verifiers will be effectively testing whether the polynomial


ˆ ˆ
_𝑄_ ( _𝑡_ ) = _𝑡_      - ( ˆ _𝑓_ ( _𝑡_ ) · ˆ _𝑔_ ( _𝑡_ ) − _ℎ_ ( _𝑡_ )) + _𝛼_ (11)


is identically zero. Whenever ˆ _𝑓_ [ˆ] - ˆ _𝑔_ ≠ _ℎ_ [ˆ], it holds that ˆ _𝑡_ - ( _𝑓_ [ˆ] ( _𝑡_ ) · ˆ _𝑔_ ( _𝑡_ ) −
_ℎ_ ( _𝑡_ )) is a non-zero polynomial. So, if ˆ _𝑓_ - _𝑔_ ˆ ≠ _ℎ_, then ˆ _𝑄_ ( _𝑡_ ) must also be
a non-zero polynomial. Note that the multiplying the term ” _𝑓_ [ˆ] - ˆ _𝑔_ − _ℎ_ [ˆ] ”
by _𝑡_ ensures that whenever this expression is non-zero, the resulting
polynomial _𝑄_ [ˆ] is guaranteed to be non-zero, even if _𝑓_ [ˆ], ˆ _𝑔_, and _ℎ_ [ˆ]
are constants, and the prover chooses _𝛼_ adversarially. Since SNIP
assumes honest verifiers, we may assume that the prover did not
know the random value _𝑟_ while generating its multiplication triple.
This implies that _𝑟_ is distributed independently of _𝛼_ which means
that we will catch a cheating prover with probability 1 − [2] _[𝑀]_ |F [+] | [1] [.]


**Beaver’s MPC Multiplication Protocol.** SNIP uses Beaver’s multiplication triples as follows. A multiplication triple is a one-timeuse triple of values ( _𝑎,𝑏,𝑐_ ) ∈ F [3], chosen at random subject to the
constraint that _𝑎_ - _𝑏_ = _𝑐_ ∈ F . In SNIP, computation, each verifier V _𝑖_
holds a share ([ _𝑎_ ] _𝑖_ _,_ [ _𝑏_ ] _𝑖_ _,_ [ _𝑐_ ] _𝑖_ ) ∈ F [3] of the triple. Using their shares
of one such triple ( _𝑎,𝑏,𝑐_ ), the verifiers can jointly evaluate shares
of the output of a multiplication gate _𝑦𝑧_ . To do so, each verifier uses


E FFeL: Ensuring Integrity For Federated Learning CCS ’22, November 7–11, 2022, Los Angeles, CA, USA



its shares [ _𝑦_ ] _𝑖_ and [ _𝑧_ ] _𝑖_ of the input wires, along with the first two
components of its multiplication triple to compute the following
values:


[ _𝑑_ ] _𝑖_ = [ _𝑦_ ] _𝑖_ −[ _𝑎_ ] _𝑖_ (12)


[ _𝑒_ ] _𝑖_ = [ _𝑧_ ] _𝑖_ −[ _𝑏_ ] _𝑖_ (13)


Each verifier V _𝑖_ then broadcasts [ _𝑑_ ] _𝑖_ and [ _𝑒_ ] _𝑖_ . Using the broadcasted shares, every verifier can reconstruct _𝑑_ and _𝑒_ and can com
pute:


[ _𝜆_ ] _𝑖_ = _𝑑𝑒_ / _𝑘_ + _𝑑_ [ _𝑏_ ] _𝑖_ + _𝑒_ [ _𝑎_ ] _𝑖_ + [ _𝑐_ ] _𝑖_ (14)


Clearly, [�] _[𝑘]_ _𝑖_ =1 [[] _[𝜆]_ []] _[𝑖]_ [=] _[ 𝑦𝑧]_ [. Thus, this step requires a round of commu-]
nication for the broadcast and three reconstructions for _𝑑_, _𝑒_ and

_𝜆_ .


For SNIPs on Shamir’s secret shares, the verifier V _𝑖_ compute the
shares ( _𝑖, 𝜆_ _𝑖_ ) where _𝜆_ _𝑖_ = _𝑑𝑒_ + _𝑑𝑏_ _𝑖_ + _𝑒𝑎_ _𝑖_ + _𝑐_ _𝑖_ which gives _𝑦𝑧_ ←
SS.recon(( _𝑖, 𝜆_ _𝑖_ )).


As mentioned in Sec. 6, we can leverage the multiplicativity of
Shamir’s secret shares to generate _𝜆_ _𝑖_ for client C _𝑖_ locally. Specifically, each client can locally multiply the shares ( _𝑗, 𝑓_ _𝑖𝑗_ ) and ( _𝑗,𝑔_ _𝑖𝑗_ )
to generate ( _𝑖,_ ( _𝑓_ _𝑗_ - _𝑔_ _𝑗_ ) _𝑖_ ) . In order to make the shares consistent, C _𝑖_
multiplies the share of ( _𝑖,ℎ_ _𝑗𝑖_ ) with ( _𝑖,𝑧_ _𝑖_ ) where _𝑧_ = 1 (these can be
generated and shared by the server S in the clear). In this way, C _𝑗_
can locally generate a share of the digest ( _𝑗,𝑑_ _𝑖𝑗_ ) that correspond
to a polynomial of degree 2 _𝑚_ . Since _𝑚_ _<_ _[𝑛]_ [−] ~~4~~ [1] [, this optimization is]

still compatible with robust reconstruction. In this way, we save
one round of communication and require only one reconstruction
for _𝜆_ _𝑖_ instead of three.


**11.2** **Complexity Analysis**


We present the complexity analysis of EIFFeL in terms of the number of clients _𝑛_, number of malicious clients _𝑚_ and data dimension
_𝑑_ (Table 1).


**Computation Cost.** Each client C _𝑖_ ’s computation cost can be broken into six components: ( 1 ) performing _𝑛_ −1 key agreements –
_𝑂_ ( _𝑛_ ) ; ( 2 ) generating proof _𝜋_ _𝑖_ for Valid( _𝑢_ _𝑖_ ) = 1 – _𝑂_ (|Valid| + M log M) [2] ;
( 3 ) creating secret shares of the update _𝑢_ _𝑖_ and the proof _𝜋_ _𝑖_ –
_𝑂_ ( _𝑚𝑛_ ( _𝑑_ + M)) [3] ; ( 4 ) verifying the validity of the received shares
– _𝑂_ ( _𝑚𝑛_ ( _𝑑_ + M) ; ( 5 ) generating proof digest for all other clients –

_𝑂_ ( _𝑛_ |Valid|) ; and ( 6 ) generating shares of the final aggregate – _𝑂_ ( _𝑛𝑑_ ) .
Assuming |Valid| is of the order of _𝑂_ ( _𝑑_ ), the overall computation
complexity of each client C _𝑖_ is _𝑂_ ( _𝑚𝑛𝑑_ ) .
The server S ’s computation costs can be divided into three parts: ( 1 )
verifying the validity of the flagged shares – _𝑂_ ( _𝑚𝑑_ min( _𝑛,𝑚_ [2] )) ; ( 2 )
verifying the proof digest for all clients – _𝑂_ ( _𝑛_ [2] log [2] _𝑛_ log log _𝑛_ ) ; and
( 3 ) computing the final aggregate – _𝑂_ ( _𝑑𝑛_ log [2] _𝑛_ log log _𝑛_ ) . Hence, the
total computation complexity of the server is _𝑂_ [�] ( _𝑛_ + _𝑑_ ) _𝑛_ log [2] _𝑛_ log log _𝑛_
+ _𝑚𝑑_ min( _𝑛,𝑚_ [2] ) [�] .


**Communication Cost.** The communication cost of each client C _𝑖_
has seven components: ( 1 ) exchanging keys with all other clients –
_𝑂_ ( _𝑛_ ) ; ( 2 ) receiving Valid(·) – _𝑂_ (|Valid|) ; ( 3 ) sending encrypted secret
shares and check strings for all other clients – _𝑂_ ( _𝑛_ ( _𝑑_ + M) + _𝑚𝑑_ ) ;


2 We use standard discrete FFT for all polynomial operations [38].
3 This uses the fact that the Lagrange coefficients can be pre-computed [58].



( 4 ) receiving encrypted secret shares and check strings from all
other clients – _𝑂_ ( _𝑛_ ( _𝑑_ + M) + _𝑚𝑛𝑑_ ) ; ( 5 ) sending proof digests for every other client – _𝑂_ ( _𝑛_ ) ; ( 6 ) receiving the list of corrupt clients C
– _𝑂_ ( _𝑚_ ) ; and ( 7 ) sending the final aggregate – _𝑂_ ( _𝑑_ ) . Thus, the communication complexity for every client is _𝑂_ ( _𝑚𝑛𝑑_ ) .
The servers communication costs include: ( 1 ) sending the validation
predicate – _𝑂_ (|Valid|) ; ( 2 ) receiving check strings and secret shares
from flagged clients – _𝑂_ ( _𝑚𝑑_ min( _𝑛,𝑚_ [2] )) ; ( 3 ) receiving proof digests
– _𝑂_ ( _𝑛_ [2] ) ; ( 4 ) sending the list of malicious clients – _𝑂_ ( _𝑚_ ) ; and ( 5 ) receiving the shares of the final aggregate – _𝑂_ ( _𝑛𝑑_ ) . Hence, the overall
communication complexity of the server is _𝑂_ ( _𝑛_ [2] + _𝑚𝑑_ min( _𝑛,𝑚_ [2] )) .
The total number of one-way communication is 12 and 9 for the
clients and server, respectively, _independent_ of the complexity of
the validation predicate.


**11.3** **Proof for Lemma 3**


Proof. In Round 3, the proof corresponding to a client C _𝑖_ is verified
iff it has submitted valid shares for the _𝑛_ − _𝑚_ − 1 honest clients C _𝐻_ \C _𝑖_ .
This is clearly true if C _𝑖_ is honest. If C _𝑖_ is malicious, _i.e._, it submitted
at least one invalid share:


- _Case 1:_ |Flag[ _𝑖_ ]| ≥ _𝑚_ + 1 . It is clear that C _𝑖_ has submitted an invalid
share to at least one honest client and, hence, is removed from
the rest of the protocol.

- _Case 2:_ |Flag[ _𝑖_ ]| ≤ _𝑚_ . All honest clients in C _𝐻_ will be flagging C _𝑖_ .
Hence, C _𝑖_ either has to submit the corresponding valid shares or
be removed from the protocol.


Given _𝑛_ − _𝑚_ − 1 valid shares, using Fact 2, we know that EIFFeL reconstructs the proof summary for C _𝑖_ correctly. Eq. 5 then follows
from the soundness property of SNIP. 

**11.4** **Proof for Lemma 5**


Proof. In Round 2, observe that the shares ( _𝑗,𝑢_ _𝑖𝑗_ ) _,_ ( _𝑗, 𝜋_ _𝑖𝑗_ ) for each
client C _𝑗_ ∈C \ _𝑖_ are encrypted with the pairwise secret key and
distributed. Hence, a collusion of _𝑚_ malicious clients (and the server
S ) [4] can access _at most_ _𝑚_ shares of any honest client C _𝑖_ ∈C _𝐻_ . This
is true even in Round 3 where:


- A malicious client might falsely flag C _𝑖_ .

- No honest client in C _𝐻_ \ C _𝑖_ will flag C _𝑖_ since they would be
receiving valid shares (and their encryptions) from C _𝑖_ .

- S cannot lie about who flagged who, since everything is logged
publicly on the bulletin B.


Thus, only _𝑚_ shares of C _𝑖_ can be revealed which correspond to the
_𝑚_ malicious clients.

Since at least _𝑚_ + 1 shares are required to recover the secret, any
instantiation of the SNIP verification protocol ( _i.e._, reconstruction
of the values of _𝜎_ _𝑖_ = ( _𝑤_ _𝑖_ _[𝑜𝑢𝑡]_ _, 𝜆_ _𝑖_ ) ) requires at least one _honest_ client
to act as the verifier. Hence, at the end of Round 3, from Fact 1
and the zero-knowledge property of SNIP, the only information
revealed is that Valid( _𝑢_ _𝑖_ ) = 1. 

4 The server does not have access to any share of its own in EIFFeL.


CCS ’22, November 7–11, 2022, Los Angeles, CA, USA Roy Chowdhury et al.



**11.5** **Security Proof**


Theorem 7. _Given a public validation predicate_ _Valid_ (·) _, security_
_parameter_ _𝜅_ _, set of_ _𝑚_ _malicious clients_ C _𝑀_ _,_ ⌊ _𝑚_ _<_ _[𝑛]_ [−] ~~3~~ [1] [⌋] _[and a ma-]_

_licious server_ S _, there exists a probabilistic polynomial-time (P.P.T.)_
_simulator Sim_ (·) _such that:_


_Real_ _EIFFeL_ �{ _𝑢_ C _𝐻_ } _,_ Ω C _𝑀_ ∪S � ≡ _𝐶_ _Sim_ �Ω C _𝑀_ ∪S _,_ U _𝐻_ _,_ C _𝐻_ �


_where_ U _𝐻_ = _𝑢_ _𝑖_ _._
∑︁

C _𝑖_ ∈C _𝐻_


{ _𝑢_ C _𝐻_ } _denotes the input of all the honest clients,_ _Real_ _EIFFeL_ _denotes_
_a random variable representing the joint view of all the parties in_
_EIFFeL’s execution,_ Ω C _𝑀_ ∪S _indicates a polynomial-time algorithm_
_implementing the “next-message” function of the parties in_ C _𝑀_ ∪S _,_
_and_ ≡ _𝐶_ _denotes computational indistinguishability._


Proof. We prove the theorem by a standard hybrid argument. Let
Ω C _𝑀_ ∪S indicate the polynomial-time algorithm that denotes the
“next-message” function of parties in C _𝑀_ ∪S . That is, given a party
identifier _𝑐_ ∈C _𝑀_ ∪S, a round index _𝑖_, a transcript _𝑇_ of all messages
sent and received so far by all parties in C _𝑀_ ∪S, joint randomness
_𝑟_ C _𝑀_ ∪S for the corrupt parties’ execution, and access to random
oracle _𝑂_, Ω C _𝑀_ ∪S ( _𝑐,𝑖,𝑇,𝑟_ C _𝑀_ ∪S ) outputs the message for party _𝑐_
in round _𝑖_ (possibly making several queries to _𝑂_ along the way).
We note that Ω C _𝑀_ ∪S is thus effectively choosing the inputs for all
corrupt users.


We will define a simulator Sim through a series of (polynomially
many) subsequent modifications to the real execution Real EIFFeL,
so that the views of Ω C _𝑀_ ∪S in any two subsequent executions are
computationally indistinguishable.


(1) Hyb 0 : This random variable is distributed exactly as the view
of Ω C _𝑀_ ∪S in Real EIFFeL, the joint view of the parties C _𝑀_ ∪S
in a real execution of the protocol.


(2) Hyb 1 : In this hybrid, for any pair of honest clients C _𝑖_ _,_ C _𝑗_ ∈C _𝐻_,
the simulator changes the key from KA.agree( _𝑝𝑘_ _𝑗_ _,𝑠𝑘_ _𝑖_ ) to a
uniformly random key. We use Diffie-Hellman key exchange
protocol in EIFFeL. The DDH assumption [ 32 ] guarantees that
this hybrid is indistinguishable from the previous one. also be
able to break the DDH.


(3) Hyb 2 : This hybrid is identical to Hyb 1, except additionally, Sim
will abort if Ω C _𝑀_ ∪S succeeds to deliver, in round 2, a message
to an honest client C _𝑖_ on behalf of another honest client C _𝑗_,
such that ( 1 ) the message is different from that of Sim, and
( 2 ) the message does not cause the decryption to fail. Such a
message would directly violate the IND-CCA security of the
encryption scheme.


(4) Hyb 3 : In this round, for every honest party in C _𝐻_, Sim samples
_𝑠_ _𝑖_ ∈ F such that Valid( _𝑠_ _𝑖_ ) = 1 and replaces all the shares and the
check strings accordingly. This allows the server to compute
the _𝜎_ _𝑖_ = ( _𝑤_ _𝑖_ _[𝑜𝑢𝑡]_ _, 𝜆_ _𝑖_ ) such that _𝑤_ _𝑖_ _[𝑜𝑢𝑡]_ = 1 ∧ _𝜆_ _𝑖_ = 0 for all honest
clients in the same way as in the previous hybrid. An adversary
noticing any difference would break ( 1 ) the computational
discrete logarithm assumption used by the VSS [ 35 ], OR ( 2 )
the IND-CCA guarantee of the encryption scheme, OR ( 3 ) the
information theoretic perfect secrecy of Shamir’s secret sharing



scheme with threshold _𝑚_ + 1, OR ( 4 ) zero-knowledge property
of SNIP.


(5) Hyb 4 : In this hybrid, Sim uses U _𝐻_ to compute the following
polynomial. Let ( _𝑗,𝑆_ _𝑗_ ) represent the share of [�] _𝑖_ ∈C _𝐻_ _[𝑠]_ _𝑖_ [for a]
malicious client C _𝑗_ ∈C \ C _𝐻_ where _𝑠_ _𝑖_ denotes the random
input Sim had sampled for C _𝑖_ ∈C _𝐻_ in Hyb 3 . Sim performs
polynomial interpolation to find the _𝑚_ + 1-degree polynomial
_𝑝_ ∗ that satisfies _𝑝_ ∗( 0 ) = U _𝐻_ and _𝑝_ ( _𝑗_ ) = _𝑆_ _𝑗_ . Next, for all
honest client, Sim computes the share for U = U _𝐻_ + [�] C _𝑗_ ∈ _𝐶_ [¯] _[𝑢]_ _[𝑗]_
(Eq. 6) by using the polynomial _𝑝_ ∗ and the relevant messages
from Ω C _𝑀_ ∪S . Clearly, this hybrid is indistinguishable from the
previous one by the perfect secrecy of Shamir’s secret shares.
This concludes our proof.


                     

**11.6** **Additional Evaluation Results**


In this section, we provide some additional evaluation results on
model accuracy in Fig. 9. We use the same configuration as the one
reported in Sec. 7. Our observations are in line with our discussion
in Sec. 7.2.


**11.7** **Discussion Cntd.**


Here, we present additional avenues of future work for EIFFeL.


**Revealing Malicious Clients.** In our current implementation, EIFFeL publishes the (partial) list of malicious clients C [∗] . To hide the
identity of malicious clients, we could include an equal number
of honest clients in the list before publishing it, thereby providing those clients plausible deniability. We leave more advanced
cryptographic solutions as a future direction.


**Private Validation Predicate.** If Valid(·) contains some secrets of
the server S, we can employ multiple servers where the computation of Valid( _𝑢_ ) is done at the servers [ 28 ]. We leave a single-server
solution of this problem for future work.


**Byzantine-Robust Aggregation.** In EIFFeL, the integrity check
is done individually on each client update, independent of all other
clients. An alternative approach to compare the local model updates
of _all_ the clients (via pairwise distance/ cosine similarity) [ 14, 15, 24,
33, 34 ] and remove statistical outliers before using them to update
the global model. A general framework to support secure Byzantinerobust aggregations rules, such as above, is an interesting future
direction.


**Valid** (·) **Structure.** If Valid(·) contains repeated structures, the _𝐺_ gate technique [19] can improve efficiency.


**Complex Aggregation Rules.** EIFFeL can be used for more complex aggregation rules, such as mode, by extending SNIP with affineaggregatable encodings (AFE) [28].


**Differential Privacy.** The privacy guarantees of EIFFeL can be
enhanced by using differential privacy (DP) to reveal a _noisy_ aggregate using techniques such as [ 44 ]. Adding DP would also provide
additional robustness guarantees [64, 84].


E FFeL: Ensuring Integrity For Federated Learning CCS ’22, November 7–11, 2022, Los Angeles, CA, USA







|100|Col2|
|---|---|
|20<br>40<br>60<br>80<br><br>Test Accuracy|~~00~~<br>~~200~~<br>~~300~~<br>~~400~~<br>~~500~~|
|20<br>40<br>60<br>80<br><br>Test Accuracy|Number of Iterations|


**(d) EMNIST: Additive noise attack with**
**Zeno++ similarity validation predicate.**

|80<br>Accuracy<br>60<br>40 Test<br>20<br>100|Col2|Col3|
|---|---|---|
|~~100~~<br>20<br>40<br>60<br>80<br>Test Accuracy|~~300~~<br>~~500~~<br>~~70~~<br>Number of Iterati|~~0~~<br>~~900~~<br>ons|



**(h) CIFAR-10: Scaling attack with norm**
**bound validation predicate**



|80<br>60<br>40<br>20<br>100<br>Num|Col2|
|---|---|
|~~100~~<br>Num<br>20<br>40<br>60<br>80<br>|~~200~~<br>~~300~~<br>~~400~~<br>~~500~~<br>er of Iterations|


**(c) FMNIST: Min-Sum attack with co-**
**sine similarity validation predicate.**

|80<br>60<br>40<br>20<br>100 300<br>Num|Col2|
|---|---|
|~~100~~<br>~~300~~<br>Num<br>20<br>40<br>60<br>80<br>|~~500~~<br>~~700~~<br>~~900~~<br>er of Iterations|



**(g) CIFAR-10: Min-Max attack with**
**Zeno++ validation predicate.**



**(a) MNIST: Scaling attack with norm**
**bound validation predicate.**


**(e) EMNIST: Scaling attack with cosine**
**similarity validation predicate.**



**(b) FMNIST: Scaling attack with norm**
**bound validation predicate.**


**(f) EMNIST: Sign flip attack with norm**
**ball validation predicate.**



**(i) CIFAR-10: Sign flip attack with**
**norm ball validation predicate.**


**Figure 9: Accuracy analysis of EIFFeL continued. Test accuracy is shown as a function of the FL iteration for different datasets and attacks.**



**Scaling EIFFeL.** Our experimental results in Sec. 7 show that EIFFeL has reasonable performance for clients sizes up to 250. One way
of scaling EIFFeL for larger client sizes can be by dividing the clients
into smaller subsets of size ∼ 250 and then running EIFFeL for each
of these subsets [16].


**Towards poly-logarithmic complexity.** Currently, dominant term
in the complexity is _𝑂_ ( _𝑚𝑛𝑑_ ) which results in a _𝑂_ ( _𝑛_ [2] ) dependence
on _𝑛_ (since we consider _𝑚_ is a fraction of _𝑛_ ). This can be reduced
to _𝑂_ ( _𝑛_ log [2] _𝑛𝑑_ ) by using the techniques from [ 9 ]. Specifically, instead of having each client verify the proofs of all others (complete
graph for the verification) we can follow the exact construction
of the _𝑘_ -regular graph _𝐺_ from [ 9 ] such that _𝑘_ = _𝑂_ (log _𝑛_ ) and only
neighbors in _𝐺_ act as verifiers for each other. The exact steps are
as follows:


(1) Each client _𝐶_ _𝑖_ generates _𝑛_ shares. It sends the corresponding
shares to its _𝑘_ -neighbors (according to graph G from [9]) and
the verification can be done on them as described currently
in EIFFeL. Note that these shares follow a _𝑡_ -out-of- _𝑛_ scheme

where _𝑡_ _< 𝑘_ .


(2) _𝐶_ _𝑖_ encrypts the shares for the non-neighbors using a threshold
(denoted by _𝑡_ _𝑒𝑛𝑐_ ) fully homomorphic encryption scheme such
as BGV [ 22 ] (the threshold variant can be obtained using work
such as [ 20, 42 ]). Note that this threshold, _𝑡_ _𝑒𝑛𝑐_ _> 𝑚_ is different



from that of the secret shares. This encryption is necessary for
ensuring data privacy since for the threshold of the shares we
could have _𝑡_ _< 𝑚_ .


(3) For the aggregation step, first the clients check the validity of
the shares of its non-neighbors (this can be done via homomorphic multiplications as shown by Feldman [ 35 ]). Next, only the
shares corresponding to the clients that (i) pass the first step
of input verification, and (ii) have valid shares, are aggregated.
Note that the shares corresponding to neighbors (for verification) can be encrypted using the public key of the encryption
scheme for this step.


(4) Each client now has the ciphertext of its share of the aggregate
(corresponding to ( _𝑖,𝑈_ _𝑖_ ) where _𝑈_ _𝑖_ = [�] _𝐶_ _𝑗_ ∈ _𝐶_ \ _𝐶_ ∗ _[𝑢]_ _𝑗𝑖_ [in the cur-]
rent EIFFeL protocol) which is sent to the server. The server
performs the reconstruction directly on these ciphertexts (using
their homomorphic property) and obtains the ciphertext of the
final aggregate. This can then be decrypted with the help of the
clients to obtain the final aggregate in the clear.


