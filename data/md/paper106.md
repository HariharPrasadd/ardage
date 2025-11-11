# S I R NN : A Math Library for Secure RNN Inference



Deevashwer Rathee _[∗]_


Microsoft Research

deevashwer@berkeley.edu


Divya Gupta
Microsoft Research

divya.gupta@microsoft.com



Mayank Rathee _[∗]_

Microsoft Research

mayankr@berkeley.edu



Rahul Kranti Kiran Goli

Microsoft Research

t-grahulk@microsoft.com



Rahul Sharma

Microsoft Research

rahsha@microsoft.com



Nishanth Chandran

Microsoft Research

nichandr@microsoft.com



Aseem Rastogi
Microsoft Research

aseemr@microsoft.com



_**Abstract**_ **— Complex machine learning (ML) inference algo-**
**rithms like recurrent neural networks (RNNs) use standard**
**functions from math libraries like exponentiation, sigmoid, tanh,**
**and reciprocal of square root. Although prior work on secure 2-**
**party inference provides specialized protocols for convolutional**
**neural networks (CNNs), existing secure implementations of**
**these math operators rely on generic 2-party computation (2PC)**
**protocols that suffer from high communication. We provide new**
**specialized 2PC protocols for math functions that crucially rely**
**on lookup-tables and mixed-bitwidths to address this perfor-**
**mance overhead; our protocols for math functions communicate**
**up to** 423 _×_ **less data than prior work. Some of the mixed bitwidth**
**operations used by our math implementations are (zero and**
**signed) extensions, different forms of truncations, multiplication**
**of operands of mixed-bitwidths, and digit decomposition (a**
**generalization of bit decomposition to larger digits). For each**
**of these primitive operations, we construct specialized 2PC**
**protocols that are more communication efficient than generic**
**2PC, and can be of independent interest. Furthermore, our math**
**implementations are numerically precise, which ensures that the**
**secure implementations preserve model accuracy of cleartext. We**
**build on top of our novel protocols to build S** **I** **R** **NN** **, a library**
**for end-to-end secure 2-party DNN inference, that provides the**
**first secure implementations of an RNN operating on time series**
**sensor data, an RNN operating on speech data, and a state-**
**of-the-art ML architecture that combines CNNs and RNNs for**
**identifying all heads present in images. Our evaluation shows that**
**S** **I** **R** **NN** **achieves up to three orders of magnitude of performance**
**improvement when compared to inference of these models using**
**an existing state-of-the-art 2PC framework.**
_**Index Terms**_ **—privacy-preserving machine learning; secure**
**two-party computation; recurrent neural networks; math func-**
**tions; mixed-bitwidths; secure inference**


I. I NTRODUCTION


In the problem of secure inference, there are two parties:
a server that holds a proprietary machine learning (ML)
model and a client that holds a private input. The goal is
for the client to learn the prediction that the model provides
on the input, with the server learning nothing about the
client’s input and the client learning nothing about the server’s
model beyond what can be deduced from the prediction itself.
Theoretically, this problem can be solved by generic secure
2-party computation (2PC) [49], [115]. Recently, this area
has made great strides with the works of [5], [10], [17]–[20],


_∗_ Equal contribution.




[25], [27], [32], [35], [37], [39], [47], [58], [64], [69], [73],

[83], [90]–[92], [99]–[102], [110] that have made it possible
to run secure inference on deep neural networks (DNNs).
Frameworks for secure inference like nGraph-HE [18], [19],
MP2ML [17], CrypTFlow [73], [99], and SecureQ8 [37]
go one step further and can automatically compile models
trained in TensorFlow/PyTorch/ONNX to 2-party or 3-party
computation protocols secure against semi-honest adversaries.
While such systems cover the secure inference of
some famous Convolutional Neural Networks (CNNs) (e.g.
ResNet [56], DenseNet [61] and MobileNet [105]) that exclusively use simple non-linear functions such as ReLU and
Maxpool, other important architectures such as Recurrent
Neural Networks (RNNs) or architectures that combine RNNs
and CNNs [104] use math functions, such as exponentiation,
reciprocal square root, sigmoid and tanh, extensively. These
RNN-based architectures are the models of choice when deal
ing with sequential or time series data like speech [36], [59],

[112]. Hence, for widespread adoption of secure inference,
especially in the RNN application domains, a robust support
for math functions is of paramount importance.
We focus on 2-party inference secure against semi-honest
adversaries [1] . In this setting, works that implement math functions fall into three categories. First, works that develop
general purpose math libraries [9], [66] using high-degree
polynomials. Second, works that use boolean circuits to implement math functions [102]. Third, works that use ad hoc
piecewise linear approximations [83] that require developer
intervention for each dataset and each model to balance

accuracy and latency, an unacceptable ask in the context of
automated frameworks for secure inference. All of these three

approaches rely on 2PC protocols from [41], [66], [115] and
suffer from huge performance overheads.
In this work, we design math functionalities that are both
provably precise and efficiently realizable via novel 2PC
protocols that we have developed. The performance of all
2PC implementations depend critically on the _bitwidth_ . While
prior works use a uniform bitwidth for the whole inference,
our math functionalities use non-uniform (or mixed) bitwidths:


1 We relegate comparisons with works that need additional parties for
security, e.g., 3-party computation (3PC) to Section VII.


they operate in low bitwidths and go to high bitwidths only
when necessary. Hence, we have developed new protocols that
enable switching between bitwidths and operating on values
of differing bitwidths. Our 2PC protocols for math functionalities have upto 423 _×_ lower communication than prior works
(Section VI-A). We have implemented these in S I R NN [2], a
library for end-to-end DNN inference, and evaluated on RNNbased models. While we focus on math functions occuring in
RNNs, our recipe for designing math functionalities is general
and can be used in other contexts. Furthermore, our math
functionalities and non-uniform bitwidth protocols can also
be used in non-RNN contexts and are of independent interest.


_A. Results in detail_


**New approximations for math functions.** In this paper,
we provide _provably precise_ functionalities, i.e. cleartext
implementations, for exponentiation, sigmoid, tanh, and
reciprocal of square root, that have been designed to
minimize cryptographic overheads. Exponentiation is used
in RBF kernels [55], sigmoid and tanh in RNNs with
LSTM [59] and GRU [36] cells, and reciprocal square root
in L2Normalization, where a vector _u_ is scaled down to
1
a unit vector by multiplying each entry of _u_ by ~~_√_~~ _u_ _[T]_ _u_ [. In a]

sharp departure from prior work in 2PC, our functionalities
follow the well-known paradigm of using lookup tables (LUT)
to get a good initial approximation of the math function
followed by an iterative algorithm such as Goldschmidt’s
iterations [50] to improve upon this approximation. We
take inspiration from embedded systems [51], [63], [72],

[113] where the goal of minimizing memory consumption
has led to efficient low-bitwidth implementations based
on fixed-point arithmetic. Our functionalities manipulate
variables with different bitwidths to maintain precision while
using minimal bitwidths. Furthermore, we formally verify
that our functionalities provide precision guarantees similar
to those provided by standard math libraries (Section V-D).


**Novel 2PC Protocols.** We provide efficient protocols for
bitwidth switching (both extensions and truncations) and
operating on values with differing bitwidths so that our
secure implementations mimic the behavior of the cleartext
math functionalities that operate on non-uniform minimal
bitwidths. As a baseline, another option is to use existing
2PC protocols that work with a uniform bitwidth (for all
values) that is large enough to accommodate all intermediate
values, i.e., avoids integer overflows. Similar to prior
works, this would force us to work over much larger rings
such as Z 2 64 . Since the complexity of secure protocols
grows proportionally with the bitwidth used, our use of
non-uniform bitwidth leads to much more communication

efficient protocols than the na¨ıve approach of uniform
bitwidth. We consider 4 main building blocks to achieve
this: (a) Extension - to increase bitwidths, (b) Truncation

- to decrease bitwidths (and precision), (c) Multiplication


2 Read as “siren”, S I R NN stands for Secure Inference for RNNs.




- to multiply an _m_ and _n_ bit integer into an ( _m_ + _n_ )-bit
output to avoid overflows (this product is later truncated
to have the right bitwidth required for further operations),
and (d) Digit decomposition - to extract relevant substrings
(that we call _digits_ ) of the input bitstring using which table
lookups are performed. Moreover, the fixed-point cleartext
code of our benchmarks also uses non-uniform bitwidths in

linear layers such as matrix multiplications and convolutions,
and we use our protocols for efficient realizations of the same.


**Secure Inference Library.** We have implemented our protocols for math functions in a new library, called S I R NN [3],
for DNN inference. We evaluate S I R NN on three state-of-the
art models that use fixed-point arithmetic with non-uniform
bitwidths [72]. Two of the models, one for the standard
Google-30 dataset and the other for sports training, use an
RNN architecture that provides accurate analysis of time series
data [74]. For the Google-30 dataset, the task is to recognize
commands like “Yes” and “No” from speech data, whereas
the sports training model provides performance feedback to a
sportsperson from sensor readings. To the best of our knowledge, this is the first empirical evaluation of secure inference of
RNNs on time series inputs like speech and sensor readings.
While it is possible to perform this inference using generic
2PC protocols, the overheads are intractable. To evaluate this
quantitatively, we implemented our benchmarks using the
state-of-the-art ABY [41] framework and this baseline is three
orders of magnitude worse in latency and communication.

Our third model uses an architecture that combines RNNs

and CNNs for the task of finding human heads in images [104].
This model uses the reciprocal square root function that is not
supported by any of the prior works on secure inference. Additionally, it makes roughly 3 million calls to sigmoid and tanh
each. In contrast, prior works on secure inference evaluated on
models with less than 3000 calls to sigmoid/tanh [83], [102].
S I R NN can run the Heads model securely in under 7 minutes.
To summarize, we make three key contributions:


1) We provide cryptographically friendly new approximations to math functions exponential, sigmoid, tanh and reciprocal square root that are provably precise (Section V).
2) We provide novel 2PC protocols for non-uniform
bitwidths (Section IV) that realize these math functionalities efficiently (up to 423 _×_ lower communication than
prior work, Section VI-A).
3) We implement these secure implementations in the library
S I R NN that provides the first secure inference of RNNs
on speech and time series sensor data and a model
that combines RNNs and CNNs. S I R NN outperforms
state-of-the-art by three orders of magnitude in size of
benchmarks (given by number of calls to math functions),
latency and communication (Section VI-C). Furthermore,
because of the high numerical precision of our math
implementations, S I R NN has no loss in model accuracy.


3 [Implementation is available at https://github.com/mpc-msri/EzPC.](https://github.com/mpc-msri/EzPC)


int16[2][2] W = ... ; int16[2] x = ...;
int16[2][2] U; int32[2] V; int32[2][2] T; int32[2] S;
U[0][0] = W[0][0]-x[0]; U[0][1] = W[0][1]-x[1];, ...
T[0][0] = U[0][0]*U[0][0], ...
V[0] = ((T[0][0] >> 12) + (T[0][1] >> 12), ...
S[0] = exp(-V[0], 32, 12, 32, 30), ...
return sign(S[0] - S[1])


Fig. 1: Fixed-point code for SVM with RBF kernel


The rest of the paper is organized as follows. We first provide a motivating example and an overview of our technical results in Section II. After discussing the necessary background
in Section III, we provide our novel protocols in Section IV.
The math functionalities are discussed in Section V with

their formal verification in Section V-D. Section VI provides
our evaluation on microbenchmarks, i.e., math functions in
isolation (Table I & Table II), DNNs used by prior work
that use math functions (Table III), and our RNN-based
benchmarks (Table IV). Finally, we discuss other related work
in Section VII.


II. O VERVIEW


We now present an overview of our approximations for
math functions and the building block protocols required
to realize them. We begin with a motivating example of an
inference task that crucially uses math functions; this will
help us highlight concepts such as scale and bitwidth changes.


**Motivating** **example.** Support vector machines (SVMs)
are one of the most widely used classical ML algorithms.
While prior work on secure inference has used SVMs with
polynomial kernels [76], [80], [87], [98] (that helps SVMs
perform classification in exponentially large dimensions), the
more powerful and hence widely used Radial Basis Function
(RBF) kernels (that operate on infinite dimensions) [55]
crucially relies on computing exponentiations, i.e., _e_ _[x]_ _, x <_ 0.
No prior work on secure 2PC inference supports RBF.
Consider the simple task of predicting rain using a feature
vector _x ∈_ R [2], where _x_ [0] and _x_ [1] are temperature and
humidity respectively, and the output is yes ( _y_ = _−_ 1) or no
( _y_ = 1). An SVM with RBF model infers the result using



sign



_k_
� _c_ _i_ _e_ _[−][γ]_ [2] _[||][W]_ _[i]_ _[−][x][||]_ [2]
� _i_ =1 �



where the vectors _W_ _i_ _∈_ R [2] are part of the model and
_c_ _i_ _∈{−_ 1 _,_ 1 _}_ . Here, _||W_ _i_ _−_ _x||_ [2] is the square of the L2 norm
or the Euclidean distance between _W_ _i_ and _x_ . Let _k_ = 2,
_γ_ = 1, _c_ 0 = 1 and _c_ 1 = _−_ 1.


**Scales and bitwidths.** Since 2PC is much more efficient
over integers than floating-point [29], [73], automated floatto-fixed converters [14], [24], [51], [72], [89], [94] can be
used to express this model as computation over integers
using fixed-point arithmetic. In fixed-point arithmetic,
_r ∈_ R is (approximately) represented using an _ℓ_ -bit integer



_⌊r_ _·_ 2 _[s]_ _⌋_ mod 2 _[ℓ]_, where _ℓ_ is the _bitwidth_ and _s ∈_ Z is the _scale_ .
Hence, fixed-point integer _a_ with scale _s_ denotes 2 _a_ _[s]_ _[∈]_ [R][.]
Consider the fixed-point code for our example given in Figure 1 generated by a float-to-fixed converter. The code stores
the input _x_ and the model parameters _W_ as 16-bit integers
with scale 12 (scale 12 is a common setting used in several
prior works on secure inference [73], [92], [99]). To compute
the inference result, it first computes _U_ _i_ = _W_ _i_ _−x_ where _U_ has
scale 12 using standard integer subtraction. Next, it computes
_T_ = _U ⊙_ _U_, where _⊙_ is pointwise multiplication. Since _U_ has
16-bit entries, to avoid integer overflows, the entries of _T_ must
be 32-bits wide. Standard integer multiplication accumulates
the scale and hence entries in _T_ have a scale of 24. Thus, the
code right shifts the entries of _T_ by 12 to bring the scale back
to 12 and accumulate them in _V_ _i_ = _||W_ _i_ _−_ _x||_ [2] . Next, it calls
exponentiation on _negative_ inputs of bitwidth 32 and scale 12
and produces the result _S_ with bitwidth 32 and scale 30. The
final result is the sign of _c_ 0 _S_ [0] + _c_ 1 _S_ [1]. S I R NN incurs less
than 30KB of communication to run this code.

Observe that the fixed-point code in Figure 1 frequently
changes bitwidths and scales with each operation. As we
describe in Figure 3 (Section V), our math functionality
for exponential would require multiplying two 32-bit values
to compute an intermediate 64-bit result. Now, if we had
to implement Figure 1 using existing 2PC protocols, we
would be forced to use uniform bitwidth of at least 64 for

all variables. In particular, the bitwidths of _x, W, U, T, V, S_
will all be 64 instead of 16 or 32. More generally, the
requirement of a high bitwidth even in one operation,
coupled with the requirement of uniform bitwidths, raises
the bitwidths of all variables and operations throughout an
inference task, resulting in a communication blowup. In
contrast S I R NN provides novel protocols for these low-level
operations of switching bitwidth and scale and multiplying
values of small bitwidth into large bitwidth. Ensuring
that bitwidths used in secure code mimic the bitwidths

used in low-bitwidth cleartext code, is the key factor in
low communication complexity of our secure math functions.
Next we give an overview of our approximations for math
functions followed by building blocks for our protocols.


_A. Our approximations for math functions_


Our math functionalities are designed keeping cryptographic
costs in mind. We first use lookup tables (LUT) to get a
good initial approximation of the math functions and then
run an iterative algorithm such as Goldschmidt’s iterations
to improve upon this approximation. Larger LUTs lead to
more precise results. However, the communication of secure
protocol for LUTs grows linearly with size of LUT. Hence, we
need to strike a balance to obtain implementations that are both
precise and communication efficient. Thus, for exponentiation
for negative inputs, we break the input bitstring _x_ into smaller
_d_ -length substrings (via digit decomposition) that are used
to index multiple 2 _[d]_ -sized LUTs. The looked up values are
multiplied into high bit intermediate results which are then
truncated to match the specified output bitwidth and scale.


Sigmoid and tanh reduce to exponentiating negative values
and reciprocating values between 1 and 2. For the latter, Ito _et_
_al_ . [63] provide a method for initial approximation of reciprocal using an LUT. After obtaining an initial approximation
with _ℓ_ bit entries and _ℓ_ _−_ 2 bits of fractional part, we iterate
using standard Goldschmidt’s method. To make these iterations communication efficient, we run them using fixed-point
arithmetic with non-uniform bit-widths. Our implementation
for reciprocal square root is similar but requires additional
work to shift the initial input to be between 1 and 2 using the
most significant non-zero bit (MSNZB).


_B. 2PC protocols in_ S I R NN


The 2PC protocols in S I R NN are based on 4 building blocks:
(a) Extension; (b) Truncation; (c) Multiplication; and (d) Digit
decomposition. Our protocols mimic the low bitwidths used
by cleartext fixed-point code, and work over power-of-2 rings,
i.e. Z 2 _ℓ_ . Let _λ_ = 128 be the computational security parameter.
_a) Extension:_ This is used to lift values from smaller
ring Z 2 _m_ to larger ring Z 2 _n_ (i.e. _m < n_ ). Although extension has been considered in honest majority three-party
computation [67], there are no specialized 2PC protocols for
it. A natural baseline, however, is provided by Yao’s garbled
circuits [4] (GC) [115], which requires around _λ_ (4 _m_ + 2 _n_ ) bits
of communication to reconstruct and re-share. In contrast, our
protocol requires around _λm_ bits of communication, that is
roughly 6 _×_ better than GC.
_b) Truncation:_ This operation is used to reduce scale
and is often used after multiplication. We require 4 kinds of
truncation operations for _ℓ_ -bit values by _s_ bits: logical and
arithmetic right shifts (that preserve the bitwidth), truncateand-reduce (outputs the truncated value in Z 2 _ℓ−s_ ), and division
by 2 _[s]_ . State-of-the-art protocol for arithmetic right shift (ARS)
was given by [99] with communication roughly _λ_ ( _ℓ_ + _s_ ) that
can also be used for logical right shift and truncate-and-reduce.
We give a new protocol for logical/arithmetic right shift with
communication _≈_ _λℓ_, i.e., independent of _λs_ . Moreover, most
of our math functionalities require only truncate-and-reduce
that decreases both scale and bitwidth. We show how to
achieve this in only _≈_ _λ_ ( _s_ +1) bits of communication. Finally,
our fixed-point benchmarks also require a division by powerof-2 operation that is different from ARS for negative _x_ and
outputs _⌈x/_ 2 _[s]_ _⌉_ . Our protocol for this division requires roughly
4 _._ 5 _×_ less communication than GC.

_c) Multiplication:_ We consider the functionality for multiplying an _m_ -bit integer with an _n_ -bit integer to produce an
_ℓ_ = ( _m_ + _n_ )-bit output. This choice of _ℓ_ ensures that there
are no overflows. A similar functionality has been considered
in the 3-party setting [67] that extends both operands to _ℓ_
bits and then invokes existing multiplication protocols over
_ℓ_ bits. This approach can be used in 2PC setting as well
using our optimized protocols for extension (that are 6 _×_ better
than GC). We provide an alternate protocol that requires 1 _._ 5 _×_


4 Depth optimized GMW [49] has higher communication than GC for our
functionalities.



less communication than the na¨ıve approach of extend-thenmultiply.
_d) Digit Decomposition:_ This splits an _ℓ_ -bit value into
_c_ = _ℓ/d_ digits of _d_ -bits. It can be realized using GC with
communication _λ_ (6 _ℓ_ _−_ 2 _c −_ 2) bits. We propose an optimized
protocol that requires communication of _≈_ _λ_ ( _c −_ 1)( _d_ + 2)
bits, that is, roughly 5 _×_ lower than GC. We build on digit
decomposition for an efficient protocol for MSNZB required
to realize the functionality for reciprocal square root.


III. P RELIMINARIES


_A. Math functions and ULP errors_


The math functions have irrational outputs which are impossible to represent exactly in finite number of bits. When using
a finite-bit representation, like floating-point or fixed-point, the
most precise implementation is the one that generates _correctly_
_rounded_ results, i.e., the output of the implementation is a
representable number that is closest to the ideal R result.
However, because of Table maker’s dilemma, such implementations are computationally very expensive [45]. Consequently,
standard math libraries like GNU’s or Intel’s libm don’t

return the correctly rounded results.
_ULP error._ The deviation between the finite-bit output and
the exact result can be quantified in three ways: absolute error,
relative error, and “units in last place” or ULPs. The former
two have serious issues and the “most natural way to measure
rounding error is in ulps” [48]; standard math libraries use
ULPs to report the precision of their implementations [4],

[111]. To see why this is the case, observe that if _r_ is a
very small real number, then the absolute error between _r_
and _r_ _[′]_ = 2 _r_, i.e., _|r −_ _r_ _[′]_ _|_ = _|r|_, is small as well. Hence,
a low absolute error can be achieved even when every bit
of the output is incorrect. Relative error, given by _|_ _[r][−]_ _r_ _[r]_ _[′]_ _[|]_ [,]

remedies this situation and _r_ _[′]_ = 2 _r_ leads to high relative errors
irrespective of the magnitude of _r_ . However, the relative error
is undefined for _r_ = 0. ULP errors have the nice property that
they are always well-defined and don’t grow or shrink with
the magnitude of _r_ . At a high level, the ULP error [5] between
an exact real result _r_ and the library output _a_ is the number of
representable numbers between _a_ and _r_ [79], [106]. We show
an example in Figure 2.
Intel’s SVML [4] has ULP error below 4 and MKL [111]
guarantees ULP error below 1. It is important for the ULP
error to be low for reusability of the library implementations
as a low error gives the developers an assurance that the
library is producing precise results inasmuch as the underlying
representation permits.


_B. Threat Model_


We consider 2-party computation secure against a _static_
_semi-honest_ adversary running in probabilistic polynomial
time. That is, we consider a computationally bounded adversary _A_ that corrupts one of the parties at the beginning of the
protocol execution, follows the protocol specification, but tries


5 See [48] for the formal definition of ULPs.


Fig. 2: The computed result `exp` ( `x` ) is in error of 3 ULPs
from the mathematically exact result _e_ _[x]_ . Dots denote the
representable numbers.


to learn additional information about the honest party’s input.
We argue security using the simulation paradigm [26], [49],

[81]. For any function _f_ to be computed, consider following
two interactions: a real interaction where _P_ 0 and _P_ 1 interact
using the protocol specification in the presence of _A_ and the
environment _Z_ and the ideal interaction where _P_ 0 _, P_ 1 send
their inputs to the trusted functionality _F_ that computes _f_
and sends the outputs to the parties. We argue that for every
real adversary _A_, there is an ideal adversary _S_ such that no
environment _Z_ interacting externally with the adversaries can
distinguish between real and ideal interactions. Our protocols
invoke several sub-protocols and for ease of exposition we
describe them using the _hybrid model_, which is the same as
a real interaction except that the sub-protocol executions are
replaced with calls to the corresponding trusted functionalities
– protocol invoking _F_ is said to be in the _F_ -hybrid model.


_C. Notation_


Let _λ_ be computational security parameter. Uppercase
_L, M, N_ denote 2 _[ℓ]_ _,_ 2 _[m]_ _,_ 2 _[n]_, respectively. [ _k_ ] refers to the set
_{_ 0 _, . . ., k −_ 1 _}_ . **1** _{b}_ denotes the indicator function that is
1 when _b_ is true, and 0 otherwise. We use the natural oneto-one correspondence between _{_ 0 _,_ 1 _}_ _[ℓ]_ and Z _L_ . Consider the
lossless lifting operators _ζ_ _ℓ_ that maps an element of ring Z _L_
to Z and _ζ_ _ℓ,m_ for _m_ ⩾ _ℓ_ that maps an element of ring
Z _L_ to Z _M_ . For brevity, we suppress these operations when
their unambiguous use can be deduced from the context. For
an element _x ∈_ Z _L_, int( _x_ ) and uint( _x_ ) refer to the signed
and unsigned values in Z respectively, where the signed case
corresponds to the 2’s complement representation. uint( _x_ ) is
defined as _ζ_ _ℓ_ ( _x_ ) and int( _x_ ) = uint( _x_ ) _−_ MSB( _x_ ) _· L_, where
MSB( _x_ ) = **1** _{x_ ⩾ 2 _[ℓ][−]_ [1] _}_ is the most significant bit. For
_x, y ∈_ Z _L_, wrap( _x, y, L_ ) is 1 if _x_ + _y_ ⩾ _L_ over Z and 0
otherwise. Finally, consider the operator _∗_ _m_ : Z _×_ Z _→_ Z _M_
where _x ∗_ _m_ _y_ = _x · y_ mod _M_ . When one or both inputs are
from some integer ring Z _L_, we use uint() and int() to map
the element to Z.

_Fixed-Point Representation._ We encode real numbers as
elements in Z _L_ using their fixed-point representation. Fixedpoint representation in Z _L_ defines 2 variables, _ℓ_ and _s_, where _ℓ_
is the _bitwidth_, _s_ is the resolution (or, fractional part bitwidth)
referred to as the _scale_ and _ℓ_ _−_ _s_ is the bitwidth for the

integer part. A real number _x ∈_ R is encoded into its fixedpoint representation ˆ _x ∈_ Z _L_ with bitwidth _ℓ_ and scale _s_ as

ˆ
_x_ = Fix ( _x, ℓ, s_ ) = _⌊x·_ 2 _[s]_ _⌋_ mod _L_ . The reverse mappings from
fixed-point representation to reals are urt ( _ℓ,s_ ) ( _a_ ) = uint( _a_ ) _/_ 2 _[s]_



for unsigned numbers and srt ( _ℓ,s_ ) ( _a_ ) = int( _a_ ) _/_ 2 _[s]_ for signed
numbers, where division is over R.


_D. Cryptographic Primitives_


_Secret Sharing._ We use 2-out-of-2 additive secret sharing
schemes over different power-of-2 rings [16], [107]. For
_x ∈_ Z _L_, we denote its shares by _⟨x⟩_ _[ℓ]_ = ( _⟨x⟩_ _[ℓ]_ 0 _[,][ ⟨][x][⟩]_ 1 _[ℓ]_ [)][ such]
that _x_ = _⟨x⟩_ _[ℓ]_ 0 [+] _[⟨][x][⟩]_ _[ℓ]_ 1 [mod] _[ L]_ [ and] _[ P]_ _[b]_ [holds] _[ ⟨][x][⟩]_ _[ℓ]_ _b_ [for] _[ b][ ∈{]_ [0] _[,]_ [ 1] _[}]_ [.]
When _ℓ_ = 1, i.e., over Z 2, we use _⟨x⟩_ _[B]_ to denote boolean
shares. In our protocols, we write “ _P_ 0 & _P_ 1 hold _⟨x⟩_ _[ℓ]_ .” to
denote that _P_ _b_ holds _⟨x⟩_ _[ℓ]_ _b_ [for] _[ b][ ∈{]_ [0] _[,]_ [ 1] _[}]_ [.]
_Oblivious Transfer._ Consider 2-party functionality 1-out-of- _k_
oblivious transfer (OT) denoted by � _k_ 1 �-OT _ℓ_, where one party
is the sender with _k ℓ_ -bit messages _x_ 0 _, . . ., x_ _k−_ 1 _∈{_ 0 _,_ 1 _}_ _[ℓ]_

and the other party is the receiver with an index _j ∈_ [ _k_ ]. The
receiver learns _x_ _j_ as the output, and the sender learns nothing.
We realize this functionality using the OT extension protocol
from [70], which optimizes and generalizes the protocol from

[62]. Additionally, we use the 1-out-of-2 correlated OT (COT)
functionality � 21 �-COT _ℓ_, which is defined as follows: sender
inputs a correlation _x_ _∈_ Z _L_, receiver inputs a choice bit
_j ∈{_ 0 _,_ 1 _}_, and the functionality outputs a random element
_r ∈_ Z _L_ to the sender and _−r_ + _j · x_ to the receiver. We
instantiate this functionality with the COT protocol from [11].
Excluding the one-time setup cost for the base OTs, � _k_ 1 �-OT _ℓ_
and � 21 �-COT _ℓ_ require 2 _λ_ + _kℓ_ and _λ_ + _ℓ_ bits of communication,
respectively, and execute in 2 rounds [6] . For the special case of
_k_ = 2, � 21 �-OT _ℓ_ requires _λ_ + 2 _ℓ_ bits of communication [11].


_E. 2PC Functionalities_


For a 2-party functionality _F_, we say that “ _P_ 0 & _P_ 1 invoke
_F_ ( _x, y_ ) to learn _⟨z⟩_ _[ℓ]_ ” to mean that _P_ 0 with input _x_ and _P_ 1
with input _y_ invoke _F_ and learn arithmetic shares of _z_ over
Z _L_, i.e., _P_ 0 gets _⟨z⟩_ _[ℓ]_ 0 [and] _[ P]_ [1] [gets] _[ ⟨][z][⟩]_ _[ℓ]_ 1 [. We write “] _[F]_ [(] _[⟨][x][⟩]_ _[ℓ]_ [)][”]
to mean that _F_ takes _⟨x⟩_ _[ℓ]_ 0 [from] _[ P]_ [0] [and] _[ ⟨][x][⟩]_ _[ℓ]_ 1 [from] _[ P]_ [1] [. In our]
protocols, we use the following 2-party functionalities.


_Millionaires’/Wrap_ : The _ℓ_ -bit Millionaires’ functionality,
_F_ Mill _[ℓ]_ [takes as input] _[ x][ ∈{]_ [0] _[,]_ [ 1] _[}]_ _[ℓ]_ [from] _[ P]_ [0] [ and] _[ y][ ∈{]_ [0] _[,]_ [ 1] _[}]_ _[ℓ]_

from _P_ 1 and returns _⟨z⟩_ _[B]_ such that _z_ = **1** _{x < y}_ . The _ℓ_   bit wrap functionality, _F_ Wrap _[ℓ]_ [on same inputs returns] _[ ⟨][z][⟩]_ _[B]_

such that _z_ = wrap( _x, y, L_ ). Note that _F_ Wrap _[ℓ]_ [(] _[x, y]_ [) =]
_F_ Mill _[ℓ]_ [(] _[L][ −]_ [1] _[ −]_ _[x, y]_ [)][. Recently, [][99][] gave an efficient]
protocol for _F_ Mill _[ℓ]_ [with communication less than] [7] _[ λℓ]_ [+14] _[ℓ]_
bits with log _ℓ_ rounds.
AND: The functionality _F_ AND takes as input ( _⟨x⟩_ _[B]_ _, ⟨y⟩_ _[B]_ )
and returns _⟨x_ _∧_ _y⟩_ _[B]_ . _F_ AND can be realized using Beaver
bit-triples [15] and [99] gave a protocol for _F_ AND with
_λ_ + 20 or 148 bits [8] of total communication.


6 Recently, MOTION [23] gave a COT protocol with similar communication
and overall 2 rounds. However, their protocol requires only a single round
of communication assuming precomputed ROT correlations. The total round
complexity of some of our protocols can benefit from this COT.
7 For ease of exposition, we use this rough upper bound to compute an
upper bound of communication of most of our protocols.
8 The best known communication for _F_ AND is 138 bits [42], however, its
implementation isn’t available.


_Boolean to Arithmetic (B2A)_ : The _ℓ_ -bit B2A functionality,
_F_ B2A _[ℓ]_ [, takes boolean shares] _[ ⟨][x][⟩]_ _[B]_ [ and outputs arithmetic]
shares of the same value, i.e., _⟨x⟩_ _[ℓ]_ . We use the COT based
protocol from [99] with communication _λ_ + _ℓ_ bits.
_Multiplexer (MUX)_ : The _ℓ_ -bit MUX functionality, _F_ MUX _[ℓ]_ [,]
takes as input _⟨x⟩_ _[B]_ and _⟨y⟩_ _[ℓ]_ and outputs _⟨z⟩_ _[ℓ]_ such that
_z_ = _y_ if _x_ = 1 and 0 otherwise. We provide an optimized
protocol that reduces communication from 2( _λ_ +2 _ℓ_ ) [99]
to 2( _λ_ + _ℓ_ ) (see Appendix A).
_Lookup Table (LUT)_ : The LUT functionality for table
_T_ with _M_ entries of _n_ -bits each, _F_ LUT _[T,m,n]_ takes as
input _⟨x⟩_ _[m]_ and outputs _⟨z⟩_ _[n]_ such that _z_ = _T_ [ _x_ ]. It
can be realized using a single call to � _M_ 1 �-OT _n_ with
communication 2 _λ_ + _Mn_ bits [42].


IV. B UILDING B LOCK P ROTOCOLS

In this section, we describe our building block protocols
that we combine later to obtain protocols for math library
functions in Section V. Our protocols extensively use the
existing 2PC functionalities described in Section III-E. In
addition, they invoke the functionality _F_ Wrap _[ℓ]_ &All1s [that takes]
as input _x ∈{_ 0 _,_ 1 _}_ _[ℓ]_ from _P_ 0 and _y ∈{_ 0 _,_ 1 _}_ _[ℓ]_ from _P_ 1
and outputs ( _⟨w⟩_ _[B]_ _||⟨e⟩_ _[B]_ ) such that _w_ = wrap( _x, y, L_ ) and
_e_ = **1** _{_ ( _x_ + _y_ mod _L_ ) = _L −_ 1 _}_ . We show that this functionality can be realized with nearly the same cost as _F_ Wrap _[ℓ]_ [by]
making a white-box use of the protocol for _F_ Mill _[ℓ]_ [from [][99][]]
(Appendix B). The resulting protocol has log _ℓ_ rounds and
at most _λℓ_ + 14 _ℓ_ bits of communication. Below we describe
our protocols for extension, truncation, multiplication, digit
decomposition and MSB-to-wrap optimization that applies
extensively to our math functionalities.


_A. Zero Extension and Signed Extension_

Zero and signed extension functions are used to extend
the bitwidths of unsigned and signed numbers, respectively.
More precisely, for an _m_ -bit number _x ∈_ Z _M_, we define
zero extension (resp. signed extension) to _n_ -bits ( _n > m_ ) by
_y_ = ZExt( _x, m, n_ ) _∈_ Z _N_ (resp. _y_ = SExt( _x, m, n_ ) _∈_ Z _N_ ),
such that uint( _y_ ) = uint( _x_ ) (resp. int( _y_ ) = int( _x_ )) holds. In
Algorithm 1, we describe our protocol for _F_ ZExt _[m,n]_ [that takes as]
input _⟨x⟩_ _[m]_ and outputs _⟨y⟩_ _[n]_, where _y_ = ZExt( _x, m, n_ ). This
protocol requires log _m_ + 2 rounds and less than _λ_ ( _m_ + 1) +
13 _m_ + _n_ bits of communication.
Correctness of our protocol can be argued as follows: By correctness of _F_ Wrap _[m]_ [and] _[ F]_ B2A _[n][−][m]_ [, it holds that]
_w_ = wrap( _⟨x⟩_ _[m]_ 0 _[,][ ⟨][x][⟩]_ 1 _[m]_ _[, M]_ [)][ and] _[ y]_ [ =][ �] [1] _b_ =0 [(] _[⟨][x][⟩]_ _b_ _[m]_ _[−]_ _[M][ ·]_
_⟨w⟩_ _[n]_ _b_ _[−][m]_ ) mod _N_ . Over Z, _w_ = _⟨w⟩_ _[n]_ 0 _[−][m]_ + _⟨w⟩_ _[n]_ 1 _[−][m]_ _−_ 2 _[n][−][m]_ _·_
wrap( _⟨w⟩_ _[n]_ 0 _[−][m]_ _, ⟨w⟩_ _[n]_ 1 _[−][m]_ _,_ 2 _[n][−][m]_ ). Thus, _M ∗_ _n_ _w_ = _M ∗_ _n_
( _⟨w⟩_ _[n]_ 0 _[−][m]_ + _⟨w⟩_ _[n]_ 1 _[−][m]_ ). Also, over Z, _x_ = _⟨x⟩_ _[m]_ 0 [+] _[⟨][x][⟩]_ _[m]_ 1 _[−][w]_ _[·][M]_ [.]
Hence, _x_ mod _N_ = _y_ .
Our protocol for signed extension, i.e., _F_ SExt _[m,n]_ [, uses the]
following equation over Z:


int( _x_ ) = _x_ _[′]_ _−_ 2 _[m][−]_ [1], for _x_ _[′]_ = _x_ + 2 _[m][−]_ [1] mod _M._ (1)


This gives [9] SExt( _x, m, n_ ) = ZExt( _x_ _[′]_ _, m, n_ ) _−_ 2 _[m][−]_ [1] .


9 A similar relation was used in [44] for truncation.



**Algorithm 1** Zero Extension, Π _[m,n]_ ZExt [:]


**Input:** _P_ 0 & _P_ 1 hold _⟨x⟩_ _[m]_ .
**Output:** _P_ 0 & _P_ 1 get _⟨y⟩_ _[n]_ for _y_ = ZExt( _x, m, n_ ).

1: _P_ 0 & _P_ 1 invoke _F_ Wrap _[m]_ [(] _[⟨][x][⟩]_ 0 _[m]_ _[,][ ⟨][x][⟩]_ 1 _[m]_ [)][ and learn] _[ ⟨][w][⟩]_ _[B]_ [.]
2: _P_ 0 & _P_ 1 invoke _F_ B2A _[n][−][m]_ [(] _[⟨][w][⟩]_ _[B]_ [)][ and learn] _[ ⟨][w][⟩]_ _[n][−][m]_ [.]
3: For _b ∈{_ 0 _,_ 1 _}_, _P_ _b_ outputs _⟨y⟩_ _b_ _[n]_ [=] _[ ⟨][x][⟩]_ _[m]_ _b_ _[−]_ _[M][ ∗]_ _[n]_ _[⟨][w][⟩]_ _[n]_ _b_ _[−][m]_ .


As a baseline, one can use garbled circuits (GC) to realize zero and signed-extensions with communication cost of
_λ_ (4 _m_ +2 _n−_ 4) bits, i.e., roughly 6 _×_ the cost of our protocols.


_B. Truncation_


We consider four types of truncation operations for ring
Z _L_ as follows: We denote the logical and arithmetic rightshift operators by _≫_ _L_ and _≫_ _A_, respectively, whose inputs
are outputs are in Z _L_ . Next, we define TR( _x, s_ ) (Truncate &
Reduce _x_ by _s_ -bits) that takes inputs in Z _L_, drops the lower _s_ bits from the bit-representation of _x_ and outputs the truncated
value in smaller ring, Z 2 _ℓ−s_ . Additionally, our benchmarks also
require the C-style division (quotients are rounded towards 0)
where the divisor is a power-of-2.


**Algorithm 2** Logical Right Shift, Π _[ℓ,s]_ LRS [:]


**Input:** _P_ 0 & _P_ 1 hold _⟨x⟩_ _[ℓ]_ .
**Output:** _P_ 0 & _P_ 1 get _⟨x≫_ _L_ _s⟩_ _[ℓ]_ .

1: For _b ∈{_ 0 _,_ 1 _}_, _P_ _b_ parses _⟨x⟩_ _b_ _[ℓ]_ [as an] _[ ℓ]_ [-bit string] _[ u]_ _[b]_ _[||][v]_ _[b]_ [, where]
_u_ _b_ _∈{_ 0 _,_ 1 _}_ _[ℓ][−][s]_ and _v_ _b_ _∈{_ 0 _,_ 1 _}_ _[s]_ .
2: _P_ 0 & _P_ 1 invoke _F_ Wrap _[s]_ [(] _[v]_ [0] _[, v]_ [1] [)][ and learn] _[ ⟨][c][⟩]_ _[B]_ [.]
3: _P_ 0 & _P_ 1 invoke _F_ Wrap _[ℓ][−][s]_ &All1s [(] _[u]_ [0] _[, u]_ [1] [)][ and learn] _[ ⟨][d][⟩]_ _[B]_ _[||⟨][e][⟩]_ _[B]_ [.]
4: _P_ 0 & _P_ 1 invoke _F_ AND ( _⟨c⟩_ _[B]_ _, ⟨e⟩_ _[B]_ ) and learn _⟨t⟩_ _[B]_ .
5: For _b ∈{_ 0 _,_ 1 _}_, _P_ _b_ sets _⟨w⟩_ _b_ _[B]_ [=] _[ ⟨][d][⟩]_ _[B]_ _b_ _[⊕⟨][t][⟩]_ _[B]_ _b_ [.]
6: _P_ 0 & _P_ 1 invoke _F_ B2A _[ℓ]_ [(] _[⟨][c][⟩]_ _[B]_ [)][ and learn] _[ ⟨][c][⟩]_ _[ℓ]_ [.]
7: _P_ 0 & _P_ 1 invoke _F_ B2A _[s]_ [(] _[⟨][w][⟩]_ _[B]_ [)][ and learn] _[ ⟨][w][⟩]_ _[s]_ [.]
8: For _b ∈{_ 0 _,_ 1 _}_, _P_ _b_ outputs _u_ _b_ _−_ 2 _[ℓ][−][s]_ _∗_ _ℓ_ _⟨w⟩_ _b_ _[s]_ [+] _[ ⟨][c][⟩]_ _[ℓ]_ _b_ [.]


_Logical Right Shift._ In Algorithm 2, we describe our protocol
for _F_ LRS _[ℓ,s]_ [that takes as input] _[ ⟨][x][⟩]_ _[ℓ]_ [and outputs] _[ ⟨][x][≫]_ _[L]_ _[s][⟩]_ _[ℓ]_ [. The]
idea is as follows: Consider _x ∈_ Z _L_ and _⟨x⟩_ _[ℓ]_ . Also, for _b ∈_
_{_ 0 _,_ 1 _}_, let _⟨x⟩_ _[ℓ]_ _b_ [=] _[ u]_ _[b]_ _[||][v]_ _[b]_ [ where] _[ u]_ _[b]_ _[ ∈{]_ [0] _[,]_ [ 1] _[}]_ _[ℓ][−][s]_ [ and] _[ v]_ _[b]_ _[ ∈]_
_{_ 0 _,_ 1 _}_ _[s]_ . Then, it can be shown that _x≫_ _L_ _s_ = _u_ 0 + _u_ 1 _−_ 2 _[ℓ][−][s]_ _·_
wrap( _⟨x⟩_ _[ℓ]_ 0 _[,][ ⟨][x][⟩]_ 1 _[ℓ]_ _[, L]_ [)+][wrap][(] _[v]_ [0] _[, v]_ [1] _[,]_ [ 2] _[s]_ [)][ [][21][]. A simple protocol]
for _F_ LRS _[ℓ,s]_ [computes shares of wrap terms over] _[ ℓ]_ [-bits and] _[ s]_ [-]
bits separately. We further optimize this protocol using the
following lemma (proof appears in Appendix C):


**Lemma 1.** _Let x ∈_ Z _L_ _, ⟨x⟩_ _[ℓ]_ _be shares of x and for b ∈{_ 0 _,_ 1 _},_
_⟨x⟩_ _[ℓ]_ _b_ [=] _[ u]_ _[b]_ _[||][v]_ _[b]_ _[, where][ u]_ _[b]_ _[ ∈{]_ [0] _[,]_ [ 1] _[}]_ _[ℓ][−][s]_ _[ and][ v]_ _[b]_ _[ ∈{]_ [0] _[,]_ [ 1] _[}]_ _[s]_ _[. Define]_
_c_ = wrap( _v_ 0 _, v_ 1 _,_ 2 _[s]_ ) _, d_ = wrap( _u_ 0 _, u_ 1 _,_ 2 _[ℓ][−][s]_ ) _, e_ = **1** _{u_ 0 +
_u_ 1 mod 2 _[ℓ][−][s]_ = 2 _[ℓ][−][s]_ _−_ 1 _} and w_ = wrap( _⟨x⟩_ _[ℓ]_ 0 _[,][ ⟨][x][⟩]_ 1 _[ℓ]_ _[, L]_ [)] _[, then]_
_it holds that w_ = _d ⊕_ ( _c ∧_ _e_ ) _._


Using this lemma, our protocol only uses wrap computations
over _ℓ_ _−_ _s_ and _s_ bits and a call to _F_ AND functionality. As
another optimization, while invoking _F_ B2A on shares of _w_,
we go to arithmetic shares over Z 2 _[s]_ (and not Z _L_ ). Overall
communication cost is less than _λ_ ( _ℓ_ + 3) + 15 _ℓ_ + _s_ + 20 and


rounds required are log _ℓ_ + 3.


_Arithmetic Right Shift._ Our protocol for _F_ ARS _[ℓ,s]_ [that outputs]
_⟨x≫_ _A_ _s⟩_ _[ℓ]_ builds upon _F_ LRS _[ℓ,s]_ using the relation [44]:
_x≫_ _A_ _s_ = _x_ _[′]_ _≫_ _L_ _s −_ 2 _[ℓ][−][s][−]_ [1], where _x_ _[′]_ = _x_ + 2 _[ℓ][−]_ [1] . Hence, it
has the same cost as Π _[ℓ,s]_
LRS [. Prior state-of-the-art protocol for]
arithmetic right shift is from CrypTFlow2 [99] that runs in
log _ℓ_ +2 rounds with communication _λ_ ( _ℓ_ + _s_ +2)+19 _ℓ_ +14 _s_
bits. Note that unlike our protocol, its communication grows
multiplicatively in _λ_ with both _ℓ_ and _s_ .


_Truncate and Reduce._ Many of our protocols can benefit
from truncate and reduce to the smaller ring over
logical/arithmetic right shift operations that output shares
in the original ring. At a high level, our protocol for
_F_ TR _[ℓ,s]_ [that outputs] _[ ⟨]_ [TR][(] _[x, s]_ [)] _[⟩]_ _[ℓ][−][s]_ [ is as follows: Using the]
above notation, TR( _x, s_ ) = _u_ 0 + _u_ 1 + wrap( _v_ 0 _, v_ 1 _,_ 2 _[s]_ ).
Hence, we can skip the computation of shares of _w_,
i.e., steps 3–7 can be skipped. Overall communication is
_λ_ ( _s_ + 1) + _ℓ_ + 13 _s_ bits. The best solution using prior
techniques is: TR( _x, s_ ) = ( _x≫_ _A_ _s_ ) mod 2 _[ℓ][−][s]_, which would
incur the same cost as the state-of-the-art ARS protocol [99],
i.e., _λ_ ( _ℓ_ + _s_ + 2) + 19 _ℓ_ + 14 _s_ bits.


_Division by power-of-2._ In addition to arithmetic right shift,
the fixed-point code for ML benchmarks require C-style division by power-of-2 to preserve model accuracy. Consider
the functionality _F_ DivPow2 _[ℓ,s]_ [that takes] _[ ⟨][x][⟩]_ _[ℓ]_ [as input and outputs]
_⟨z⟩_ _[ℓ]_ such that _z_ = _⌈_ int( _x_ ) _/_ 2 _[s]_ _⌉_ mod _L_ for _z <_ 0 and
_z_ = _⌊_ int( _x_ ) _/_ 2 _[s]_ _⌋_ mod _L_ for _z_ ⩾ 0. We give an overview of
our protocol in Appendix C that requires roughly _λ_ ( _ℓ_ +2 _s_ +4)
bits of communication. To the best of our knowledge, no
prior work explicitly builds a protocol for this functionality.
A garbled circuits implementation, costs _λ_ (8 _ℓ_ + 2 _s −_ 6) bits.


_C. Multiplication with non-uniform bitwidths_


Our machine learning models as well as math library
functions (see Section V) use multiplication operation with
operands of different bit-widths that outputs a value in the
larger ring. Below, we describe these functions and their
protocols for both the unsigned and the signed case.


_Unsigned Multiplication with non-uniform bitwidths._ Consider
the functionality _F_ UMult _[m,n]_ [that takes] _[ ⟨][x][⟩]_ _[m]_ [ and] _[ ⟨][y][⟩]_ _[n]_ [ as input and]
returns _⟨z⟩_ _[ℓ]_, where _z_ = _x ∗_ _ℓ_ _y_, for _ℓ_ = _m_ + _n_ . In contrast,
all prior works on secure inference [64], [83], [90], [92], [99],

[102], use _m_ = _n_ = _ℓ_ . A na¨ıve way to realize this functionality
is to first extend both the inputs to _ℓ_ -bits and then use standard
multiplication, i.e., multiply _F_ ZExt _[m,ℓ]_ [(] _[⟨][x][⟩]_ _[m]_ [)][ and] _[ F]_ ZExt _[n,ℓ]_ [(] _[⟨][y][⟩]_ _[n]_ [)]
using existing protocols for uniform bit-widths. We give a
new custom protocol for multiplying values of non-uniform
bitwidths that beats this na¨ıve approach by roughly 1 _._ 5 _×_ . Our
protocol builds on the functionality _F_ CrossTerm _[m,n]_ [:][ Z] _[M]_ _[ ×]_ [ Z] _[N]_ _[ →]_
Z _L_ _×_ Z _L_ that is defined as _F_ CrossTerm _[m,n]_ [(] _[x, y]_ [) =] _[ ⟨][z][⟩]_ _[ℓ]_ [, where]
_z_ = _x ∗_ _ℓ_ _y_ . We describe our protocol for _F_ CrossTerm _[m,n]_ [in]
Appendix D1 that carefully uses � 21 �-COT (to minimize overall



**Algorithm 3** Unsigned Multiplication, Π _[m,n]_ UMult [:]


**Input:** _P_ 0 & _P_ 1 hold _⟨x⟩_ _[m]_ and _⟨y⟩_ _[n]_ .
**Output:** _P_ 0 & _P_ 1 get _⟨z⟩_ _[ℓ]_, where _z_ = _x ∗_ _ℓ_ _y_ and _ℓ_ = _m_ + _n_ .

1: For _b ∈{_ 0 _,_ 1 _}_, let _x_ _b_ = _⟨x⟩_ _b_ _[m]_ [and] _[ y]_ _[b]_ [=] _[ ⟨][y][⟩]_ _[n]_ _b_ [.]
2: _P_ 0 and _P_ 1 invoke the following functionalities.
3: _F_ CrossTerm _[m,n]_ [(] _[x]_ [0] _[, y]_ [1] [)][ and learn] _[ ⟨][c][⟩]_ _[ℓ]_ [.]
4: _F_ CrossTerm _[n,m]_ [(] _[y]_ [0] _[, x]_ [1] [)][ and learn] _[ ⟨][d][⟩]_ _[ℓ]_ [.]
5: _F_ Wrap _[m]_ [(] _[x]_ [0] _[, x]_ [1] [)][ to learn] _[ ⟨][w]_ _[x]_ _[⟩]_ _[B]_ [.]
6: _F_ Wrap _[n]_ [(] _[y]_ [0] _[, y]_ [1] [)][ to learn] _[ ⟨][w]_ _[y]_ _[⟩]_ _[B]_ [.]
7: _F_ MUX _[m]_ [(] _[⟨][w]_ _[y]_ _[⟩]_ _[B]_ _[,][ ⟨][x][⟩]_ _[m]_ [)][ to learn] _[ ⟨][g][⟩]_ _[m]_ [.]
8: _F_ MUX _[n]_ [(] _[⟨][w]_ _[x]_ _[⟩]_ _[B]_ _[,][ ⟨][y][⟩]_ _[n]_ [)][ to learn] _[ ⟨][h][⟩]_ _[n]_ [.]

9: _P_ _b_ outputs _x_ _b_ _∗_ _ℓ_ _y_ _b_ + _⟨c⟩_ _b_ _[ℓ]_ [+] _[ ⟨][d][⟩]_ _[ℓ]_ _b_ _[−]_ _[N][ ∗]_ _[ℓ]_ _[⟨][g][⟩]_ _[m]_ _b_ _[−]_ _[M][ ∗]_ _[ℓ]_ _[⟨][h][⟩]_ _[n]_ _b_ [for]
_b ∈{_ 0 _,_ 1 _}_ .


communication) similar to the techniques of generating Beaver
triples [15]. The communication complexity of this protocol
is _µ_ ( _λ_ + _µ/_ 2 + 1 _/_ 2) + _mn_, where _µ_ = min( _m, n_ ).
By definition of _∗_ _ℓ_, we wish to compute uint( _x_ ) _·_
uint( _y_ ) mod _L_, where _ℓ_ = _m_ + _n_ . Let _x_ = _x_ 0 + _x_ 1 mod _M_
and _y_ = _y_ 0 + _y_ 1 mod _N_ . Algorithm 3 gives our protocol for
_F_ UMult _[m,n]_ [that builds on the following: Over][ Z][,]


uint( _x_ ) _·_ uint( _y_ ) = ( _x_ 0 + _x_ 1 _−_ 2 _[m]_ _w_ _x_ ) _·_ ( _y_ 0 + _y_ 1 _−_ 2 _[n]_ _w_ _y_ )


= _x_ 0 _y_ 0 + _x_ 1 _y_ 1 + _x_ 0 _y_ 1 + _x_ 1 _y_ 0
_−_ 2 _[m]_ _w_ _x_ _y −_ 2 _[n]_ _w_ _y_ _x −_ 2 _[ℓ]_ _w_ _x_ _w_ _y_ _,_ (2)


where _w_ _x_ = wrap( _x_ 0 _, x_ 1 _, M_ ) and _w_ _y_ = wrap( _y_ 0 _, y_ 1 _, N_ ).
Taking a mod _L_, removes the last term. In the protocol, party
_P_ _b_ computes _x_ _b_ _y_ _b_ as ( _x_ _b_ _∗_ _ℓ_ _y_ _b_ ) locally and invokes _F_ CrossTerm _[m,n]_
to compute shares of cross-terms _x_ _b_ _y_ 1 _−b_ . Wraps are computed
using _F_ Wrap and multiplied to values using _F_ MUX .
The communication complexity of our protocol is roughly
_λ_ (3 _µ_ + _ν_ ) + _µ_ ( _µ_ + 2 _ν_ ) + 16( _m_ + _n_ ) where _µ_ = min( _m, n_ )
and _ν_ = max( _m, n_ ). In contrast, communication complexity
of na¨ıve approach of extend-then-multiply that uses our optimized protocols for extension is roughly 3 _λ_ ( _µ_ + _ν_ )+( _m_ + _n_ ) [2] +
15( _m_ + _n_ ), i.e., roughly 1 _._ 5 _×_ more than our new protocol.
We note that the same ideas also work for the setting
_ℓ< m_ + _n_ by using an appropriate protocol for _F_ CrossTerm _[m,n,ℓ]_
with specific value of _ℓ_ . Similarly, we define the multiplication
functionality _F_ UMult _[m,n,ℓ]_ which internally invokes _F_ CrossTerm _[m,n,ℓ]_ [,]
where the additional superscript denotes the bitwidth of the
output. Our protocols for math library functions also uses this
setting for better efficiency.


_Signed Multiplication with non-uniform bitwidths._ Consider
the functionality _F_ SMult _[m,n]_ [that takes] _[ ⟨][x][⟩]_ _[m]_ [ and] _[ ⟨][y][⟩]_ _[n]_ [ as input]
and returns _⟨z⟩_ _[ℓ]_, where _z_ = int( _x_ ) _∗_ _ℓ_ int( _y_ ), for _ℓ_ = _m_ + _n_ .
Let _x_ _[′]_ = _x_ + 2 _[m][−]_ [1] mod _M, y_ _[′]_ = _y_ + 2 _[n][−]_ [1] mod _N_ such that
_x_ _[′]_ = _x_ _[′]_ 0 [+] _[x]_ 1 _[′]_ [mod] _[ M]_ [ and] _[ y]_ _[′]_ [ =] _[ y]_ 0 _[′]_ [+] _[y]_ 1 _[′]_ [mod] _[ N]_ [. Our protocol]
for _F_ SMult _[m,n]_ [builds on the following equations over][ Z][:]


int( _x_ ) _·_ int( _y_ ) = ( _x_ _[′]_ _−_ 2 _[m][−]_ [1] ) _·_ ( _y_ _[′]_ _−_ 2 _[n][−]_ [1] ) from Eq. 1

= _x_ _[′]_ _· y_ _[′]_ _−_ 2 _[m][−]_ [1] _y_ _[′]_ _−_ 2 _[n][−]_ [1] _x_ _[′]_ + 2 _[m]_ [+] _[n][−]_ [2]

= _x_ _[′]_ _· y_ _[′]_ _−_ 2 _[m][−]_ [1] ( _y_ 0 _[′]_ [+] _[ y]_ 1 _[′]_ _[−]_ [2] _[n]_ _[w]_ _[y]_ _[′]_ [)]

_−_ 2 _[n][−]_ [1] ( _x_ _[′]_ 0 [+] _[ x]_ _[′]_ 1 _[−]_ [2] _[m]_ _[w]_ _[x]_ _[′]_ [) + 2] _[m]_ [+] _[n][−]_ [2] _[,]_


where _w_ _x_ _′_ = wrap( _x_ _[′]_ 0 _[, x]_ 1 _[′]_ _[, M]_ [)] _[, w]_ _[y]_ _[′]_ [ =][ wrap][(] _[y]_ 0 _[′]_ _[, y]_ 1 _[′]_ _[, N]_ [)][.]
In the protocol, parties can compute the shares of _x_ _[′]_ _, y_ _[′]_

locally. All terms in the final expression can be computed and
added locally except _z_ 1 = _x_ _[′]_ _y_ _[′]_ and _z_ 2 = 2 _[ℓ][−]_ [1] ( _w_ _x_ _′_ + _w_ _y_ _′_ ).
Since the final expression needs to be computed mod _L_, we
can compute shares of _z_ 1 in Z _L_ using a call to Π _[m,n]_ UMult [. We]
piggyback the computation of boolean shares of _w_ _x_ _[′]_ and _w_ _y_ _[′]_
on Π _[m,n]_ UMult [, which already computes them in steps][ 5][&][6][. Note]
that 2 _[ℓ][−]_ [1] _w_ _x_ _′_ = 2 _[ℓ][−]_ [1] ( _⟨w_ _x_ _′_ _⟩_ _[B]_ 0 [+] _[ ⟨][w]_ _[x]_ _[′]_ _[⟩]_ _[B]_ 1 _[−]_ [2] _[⟨][w]_ _[x]_ _[′]_ _[⟩]_ _[B]_ 0 _[⟨][w]_ _[x]_ _[′]_ _[⟩]_ _[B]_ 1 [)]
and taking a mod _L_ gets rid of the last term. Hence,
2 _[ℓ][−]_ [1] ( _⟨w_ _x_ _′_ _⟩_ _[B]_ + _⟨w_ _y_ _′_ _⟩_ _[B]_ ) are correct arithmetic shares of _z_ 2
in Z _L_ . Thus, we can do signed multiplication with a single
call to Π _[m,n]_ UMult [and no additional cost.]
We also consider the signed-multiplication functionality
_F_ SMult _[m,n,ℓ]_ [, where the output bitwidth] _[ ℓ< m]_ [ +] _[ n]_ [. The above]
discussion on signed-multiplication holds in this case as well,
and thus, Π _[m,n,ℓ]_ SMult [has the same cost as][ Π] _[m,n,ℓ]_ UMult [.]


_Matrix Multiplication and Convolutions._ Two commonly used
operations in machine learning are matrix multiplications and
convolutions that build on element-wise multiplications.
Consider matrix multiplication of _A_ _∈_ Z _[d]_ _M_ [1] _[×][d]_ [2] and
_B ∈_ Z _[d]_ _N_ [2] _[×][d]_ [3], where we would like to use our protocol
for _F_ UMult _[m,n]_ [. Now, each element in the output product matrix]
is a result of _d_ 2 multiplications and _d_ 2 _−_ 1 additions and
even when the result of multiplication is stored in the larger
ring Z _L_, _ℓ_ = _m_ + _n_, the value can overflow due to additions.
One way to avoid this overflow is to extend the result of
element-wise products by _e_ = _⌈_ log _d_ 2 _⌉_ bits and then do
the additions. However, this method is quite expensive as
the number of extensions needed would be _d_ 1 _d_ 2 _d_ 3 . We
significantly reduce this cost as follows: Since the cost of
_F_ CrossTerm depends on the smaller of the two bitwidths,
we extend the values in the matrix of larger bitwidth by
_e_ bits. Then we perform the matrix multiplications into
Z 2 _m_ + _n_ + _e_, ensuring that there are no overflows. Moreover,
similar to the OT-based matrix multiplication from prior
works [92], [99], we also exploit the multi-use of input
matrix elements to optimize the cost of computing (matrix)
cross-terms in our protocol. Our protocol has communication
complexity roughly _λ_ (3 _d_ 1 _d_ 2 ( _m_ + 2) + _d_ 2 _d_ 3 ( _n_ + 2)) +
_d_ 1 _d_ 2 _d_ 3 �(2 _m_ + 4)( _n_ + _e_ ) + _m_ [2] + 5 _m_ � bits for _m_ ⩽ _n_
ignoring lower order terms. We describe our protocol formally
in Appendix D2 along with exact communication complexity.
Above ideas easily extend to computing convolutions as well.


_Multiply and Truncate._ In most of our protocols, we first
invoke _F_ SMult _[m,n,ℓ]_ [followed by] _[ F]_ TR _[ℓ,s]_ [, where] _[ ℓ]_ [⩽] _[m]_ [ +] _[ n]_ [. Hence,]
for ease of exposition, we define the functionality _F_ SMultTR _[m,n,ℓ,s]_ [for]
signed multiplication and truncate-reduce that takes _⟨x⟩_ _[m]_ and
_⟨y⟩_ _[n]_ as input and returns _⟨z_ _[′]_ _⟩_ _[ℓ][−][s]_ such that _z_ = int( _x_ ) _∗_ _ℓ_ int( _y_ )
and _z_ _[′]_ = TR( _z, s_ ).


_D. Digit Decomposition and MSNZB_

We consider the functionality _F_ DigDec _ℓ,{d_ _i_ _}_ _i∈_ [ _c_ ] that decomposes
an _ℓ_ -bit number into _c_ sub-strings or digits of lengths _{d_ _i_ _}_ .



More formally, _F_ DigDec _ℓ,{d_ _i_ _}_ _i∈_ [ _c_ ] takes _⟨x⟩_ _[ℓ]_ as input and outputs
_⟨z_ _c−_ 1 _⟩_ _[d]_ _[c][−]_ [1] _, . . ., ⟨z_ 0 _⟩_ _[d]_ [0] such that _x_ = _z_ _c−_ 1 _|| . . . ||z_ 0 .
For an _ℓ_ -bit integer _x_, MSNZB( _x_ ) refers to the index of the
most significant non-zero-bit. That is, MSNZB( _x_ ) = _k ∈_ [ _ℓ_ ],
if _x_ _k_ = 1 and _x_ _j_ = 0 for all _j > k_ . Consider the functionality
_F_ MSNZB _[ℓ]_ [that takes as input] _[ ⟨][x][⟩]_ _[ℓ]_ [and outputs] _[ {⟨][z]_ _[i]_ _[⟩]_ _[B]_ _[}]_ _[i][∈]_ [[] _[ℓ]_ []] [ such]
that _z_ _i_ = 1 if MSNZB( _x_ ) = _i_ and 0 otherwise.
We describe the protocols for _F_ DigDec _ℓ,{d_ _i_ _}_ _i∈_ [ _c_ ] and _F_ MSNZB _[ℓ]_ [in]
Appendix E and F, respectively.


_E. MSB-to-Wrap Optimization_


Our protocols above for extension, truncation and multiplication make use of the following step: Parties _P_ 0 _, P_ 1 hold
_⟨x⟩_ _[ℓ]_ and compute _⟨w⟩_ _[B]_, where _w_ = wrap( _⟨x⟩_ _[ℓ]_ 0 _[,][ ⟨][x][⟩]_ 1 _[ℓ]_ _[, L]_ [)][.]
This is either computed through an explicit call to _F_ Wrap _[ℓ]_
(e.g., extension and multiplication) or computed via wrap of
lower and upper bits (e.g., truncation). We show that shares
of _w_ can be computed with much less communication and
rounds if the parties either know the _m_ _x_ = MSB( _x_ ) in the
clear or shared form. The MSB refers to the most significant
bit of a number. In our math library implementations in
Section V, this condition is true for almost all invocations.
For instance, in exponential, when multiplying the values
from multiple LUTs, we know that all operands are positive,
i.e., MSB of all inputs to multiplication is 0. We call this
optimization _MSB-to-Wrap_ and the idea is as follows: We can
write _w_ = ((1 _⊕_ _m_ _x_ ) _∧_ ( _m_ 0 _⊕_ _m_ 1 )) _⊕_ ( _m_ 0 _∧_ _m_ 1 ), where
_m_ _b_ = MSB( _⟨x⟩_ _[ℓ]_ _b_ [)][ for] _[ b][ ∈{]_ [0] _[,]_ [ 1] _[}]_ [. With this, given shares of]
_m_ _x_, boolean shares of _w_ can be computed using a single call
to � 41 �-OT 1, i.e., 2 _λ_ + 4 bits of communication and 2 rounds.
Also, when _m_ _x_ is publicly known, this can be computed using
� 21 �-OT 1, i.e., _λ_ +2 bits. The cost of our protocols with above
optimization are provided in Table V.


V. M ATH L IBRARY F UNCTIONS


In this section, we provide our cleartext implementations for
math functions exponential, sigmoid, tan hyperbolic (tanh),
and reciprocal square root as well as the protocols for the
same. Note that these functions are impossible to implement
exactly using finite-bit arithmetic, and hence, our implementations realize them approximately (Section V-D). Below, we
use the notation from Section III-C and Section III-D. For

a mathematical function _f_, we consider the functionality
_F_ _f_ _[m,s,n,s]_ _[′]_ that takes as input the shares _⟨x⟩_ _[m]_ and outputs _⟨y⟩_ _[n]_

such that srt ( _n,s_ _′_ ) ( _y_ ) _≈_ _f_ (srt ( _m,s_ ) ( _x_ )).
Our math function implementations rely on functions discussed in Section IV, and we recall some of them here.
We denote signed-extension of an _m_ -value to an _n_ -value
by SExt( _x, m, n_ ) with _n > m_ . Next, we denote truncateand-reduce by _s_ -bits using TR( _x, s_ ) that takes a value _x_ of,
say, _ℓ_ -bits, drops lower _s_ bits and returns the corresponding
( _ℓ−s_ )-bit value. Finally, we use a signed multiplication where
the operands and the output can have unequal bitwidths. It
is denoted by _x ∗_ _ℓ_ _y_, where _x_ and _y_ are, say, _m_ and _n_ bit integers, respectively, and the output of multiplication is
_z_ = int( _x_ ) _·_ int( _y_ ) mod _L_ .


_A. Exponential_



Consider the math functionality _F_ rExp _[m,s,n,s]_ _[′]_ with rExp( _z_ ) =

_e_ _[−][z]_, _z ∈_ R [+] described in Figure 3. Intuitively, the correctness
of this functionality, i.e., srt ( _n,s_ _′_ ) ( _y_ ) _≈_ rExp(srt ( _m,s_ ) ( _x_ )),
relies on rExp(srt ( _m,s_ ) ( _x_ )) = rExp(2 _[d]_ [(] _[k][−]_ [1)] _[−][s]_ _x_ _k−_ 1 ) _· . . . ·_
rExp(2 _[−][s]_ _x_ 0 ). Each rExp call on the RHS can be computed
approximately using a lookup table _L_ of size 2 _[d]_ with _s_ _[′]_ +2 bit
entries of scale _s_ _[′]_ . Since the entries of the LUTs are between
0 and 1 with scale _s_ _[′]_, it is sufficient to have a bitwidth of
_s_ _[′]_ + 2. For instance, when _m_ = _n_ = 16, _d_ = 8, and _s_ _[′]_ = 14
we use two LUTs where first maps the upper 8 bits of _x_ and
second maps the lower 8 bits of _x_ . Final output is computed
by multiplying the two 16-bit looked up values from the two
LUTs into a 32-bit number followed by an appropriate truncate
and reduce operation to get 16-bit _y_ with scale 14. We formally
verify that for _m, s, n, s_ _[′]_ used in our evaluation, our choice of
_d_ ensures precise results in Section V-D.
The protocol for this functionality can be built easily relying
on the protocols described in Section IV. Step 1 can be
implemented by a call to the digit decomposition functionality,
_F_ DigDec . The LUTs in Step 2 can be looked up using _F_ LUT
(Section III-E). These _s_ _[′]_ + 2-bit values are multiplied using
a tree-based multiplication using _F_ SMultTR _[s]_ _[′]_ [+2] _[,s]_ _[′]_ [+2] _[,]_ [2] _[s]_ _[′]_ [+2] _[,s]_ _[′]_ to get an

_s_ _[′]_ + 2-bit number with scale _s_ _[′]_ in Step 3. Finally, Step 4
extends _g_ to an _n_ -bit value using _F_ SExt _[s]_ _[′]_ [+2] _[,n]_ . Table II gives our
concrete numbers and compares with prior work.



**Functionality** _F_ _h_ _[m,s,n,s]_ _[′]_ ( _⟨x⟩_ _[m]_ )



1) _⟨u⟩_ _[s]_ _[′]_ [+2] _←F_ _[m,s,s]_ _[′]_ [+2] _[,s]_ _[′]_



1) _⟨u⟩_ _[s]_ _←F_ rExp _[m,s,s]_ _[,s]_ ( _⟨x⟩_ _[m]_ ).

2) _⟨w⟩_ _[s]_ _[′]_ [+2] _←F_ Rec _[s]_ _[′]_ [+2] _[,s]_ _[′]_ ( _⟨_ 2 _[s]_ _[′]_ + _u⟩_ _[s]_



2) _⟨w⟩_ _[s]_ _[′]_ [+2] _←F_ Rec _[s]_ _[′]_ [+2] _[,s]_ _[′]_ ( _⟨_ 2 _[s]_ _[′]_ + _u⟩_ _[s]_ _[′]_ [+2] ).

3) Return SExt( _w, s_ _[′]_ + 2 _, n_ ).



Fig. 4: _The functionality F_ _h_ _[m,s,n,s]_ _[′]_ _._



**Functionality** _F_ rExp _[m,s,n,s]_ _[′]_ ( _⟨x⟩_ _[m]_ )



1) Let _x_ = _x_ _k−_ 1 _|| . . . ||x_ 0, _x_ _i_ _∈{_ 0 _,_ 1 _}_ _[d]_, _i ∈_ [ _k_ ], _dk_ = _m_ .
2) For _i ∈_ [ _k_ ], let _L_ _i_ : _{_ 0 _,_ 1 _}_ _[d]_ _→_ Z 2 _s′_ +2 s.t. _L_ _i_ ( _j_ ) =
Fix �rExp(2 _[di][−][s]_ _j_ ) _, s_ _[′]_ + 2 _, s_ _[′]_ [�] .
3) Compute _g_ = _L_ _k−_ 1 [ _x_ _k−_ 1 ] _∗_ _. . . ∗_ _L_ 0 [ _x_ 0 ], _g_ has bitwidth
_s_ _[′]_ + 2 and scale _s_ _[′]_ .
4) Return _⟨y⟩_ _[n]_ for _y_ = SExt( _g, s_ _[′]_ + 2 _, n_ ).


Fig. 3: _The functionality F_ rExp _[m,s,n,s]_ _[′]_ _for a parameter d._


_B. Sigmoid and Tanh_

Consider the math functionality _F_ sigmoid _[m,s,n,s]_ _[′]_ where

1
sigmoid( _z_ ) = 1+ _e_ _[−][z]_ [can be written as]



reciprocal of values _v_ such that 1 ⩽ srt ( _ℓ,s_ ) ( _v_ ) _<_ 2 which
is true for the case of _h_ and sigmoid.
We describe the math functionality _F_ Rec _[ℓ,s]_ [in][ Figure 5][ that]
maps inputs _v_ with bitwidth _ℓ_ and scale _s_ to outputs of same
bitwidth and scale. Since 1 ⩽ srt ( _ℓ,s_ ) ( _v_ ) _<_ 2, in Step 1, _d_ = 1.
We use the _g_ most significant bits of the fractional part to index
into the LUT _L_ rec in Step 2 whose entries are described in [63].
The initial approximation _w_ has bitwidth _s_ + 1 and scale _s_ . If
the number of Goldschmidt iterations _t_ is set to 0, then _F_ Rec
outputs initial approximation sign extended to output bitwidth,
i.e., SExt( _w, s_ + 1 _, ℓ_ ). We formally verify that for _m, s, n, s_ _[′]_

used in our evaluation, our choice of parameters for _F_ rExp and
_F_ Rec ensures precise results for _F_ sigmoid in Section V-D.
Note that this functionality crucially utilizes arithmetic over
variable bitwidth and extension/truncation operations and these
steps require our efficient protocols from Section IV. Table I
gives our concrete numbers and compares with prior work.


**Functionality** _F_ Rec _[ℓ,s]_ [(] _[⟨][v][⟩]_ _[ℓ]_ [)]
Computes the initial approximation _w_ as follows [63]:

1) _v_ = _d||e||f_, _d ∈{_ 0 _,_ 1 _}_ _[ℓ][−][s]_, _e ∈{_ 0 _,_ 1 _}_ _[g]_, _f ∈{_ 0 _,_ 1 _}_ _[s][−][g]_ .
2) _c_ 0 _||c_ 1 = _L_ rec ( _e_ ), _c_ 0 _∈{_ 0 _,_ 1 _}_ _[g]_ [+4] and _c_ 1 _∈{_ 0 _,_ 1 _}_ [2] _[g]_ [+3] .
3) _c_ 2 = SExt(( _c_ 0 _∗_ _s_ +4 _f_ ) _, s_ + 4 _, s_ + _g_ + 4).
4) _w_ _[′]_ = 2 _[s][−][g]_ [+1] _∗_ _s_ + _g_ +4 _c_ 1 _−_ _c_ 2, _w_ = TR( _w_ _[′]_ _, g_ + 3).

Goldschmidt’s method for _t_ iterations.


1) _p_ 1 = 2 _[s]_ _−_ TR( _v ∗_ 2 _s_ +2 _w, s_ ).
2) _q_ 1 = 2 _[s]_ + _p_ 1 _, a_ 1 = _q_ 1 .
3) For _i ∈{_ 2 _, . . ., t}_ do

a) _a_ _i_ = TR( _a_ _i−_ 1 _∗_ 2 _s_ +2 _q_ _i−_ 1 _, s_ ).
b) _p_ _i_ = TR( _p_ _i−_ 1 _∗_ 2 _s_ +2 _p_ _i−_ 1 _, s_ ).

c) _q_ _i_ = 2 _[s]_ + _p_ _i_ .
4) Return SExt( _a_ _t_ _, s_ + 2 _, ℓ_ ).


Fig. 5: _The functionality F_ Rec _[ℓ,s]_ _[for a parameters][ g, t][.]_



sigmoid( _z_ ) =



0 _._ 5 _,_ if _z_ = 0

1 if _z >_ 0

rExp 1+rExp ( _−_ ( _z_ _z_ ) ) _[,]_ 1+rExp1 ( _−z_ ) _[,]_ if _z <_ 0



Hence, sigmoid can be built by extending the math functionality _F_ _h_ _[m,s,n,s]_ _[′]_ such that _h_ ( _z_ ) = 1+rExp1 ( _z_ ) [,] _[ z][ ∈]_ [R] [+] [ described]

in Figure 4. This functionality calls _F_ rExp that we described
above, followed by a call to a functionality to approximate the
reciprocal that we describe next.
For computing the reciprocal, we rely on the Goldschmidt’s
algorithm [50] that iterates on an initial approximation [63].
This initial approximation requires that we only compute



_Tanh._ The math functionality _F_ Tanh _[m,s,n,s]_ _[′]_ where Tanh( _z_ ) =

_ee_ _[z][z]_ _−_ + _ee_ _[−][−][z][z]_ [= 2] _[ ·]_ [ sigmoid][(2] _[z]_ [)] _[ −]_ [1][ can be realized using] _[ F]_ sigmoid [.]


_C. Reciprocal of Square Root_


In ML, reciprocal square root is typically used to scale down
vectors _⃗u_ of large magnitude to unit vectors by dividing each
entry of the vector with ~~_√_~~ _u_ 1 _[T]_ _u_ [. The reciprocal square root]

function maps _x_ to ~~_√_~~ 1 ~~_x_~~, for _x >_ 0. If _x_ is small then to
avoid divide-by-zero errors a small public constant _ϵ_ is added
to _x_ and ~~_√_~~ _x_ 1+ _ϵ_ is computed instead. Hence, we present our

mathematical functionality _F_ rsqrt _[ℓ,s,ℓ,s]_ _[′]_ in Figure 6 for the math
function rsqrt( _z_ ) = ~~_√_~~ 1 ~~_z_~~ where _z_ ⩾ _ϵ_ .


This functionality follows a similar template of first computing an initial approximation for reciprocal square root followed
by Goldschmidt’s iterations. The initial approximation [10] requires 1 ⩽ _x <_ 2, and hence, first we perform a range
reduction to map arbitrary _x_ of the form _y.z_ to _x_ _[′]_ of the form
1 _.z_ _[′]_ that satisfies this constraint. This requires computing the
most significant non-zero bit (MSNZB) of _x_ (Step 1). Note
that MSNZB( _x_ ) = _k ∈_ [ _ℓ_ ] if _x_ _k_ = 1 and all _x_ _i_ = 0 for all
_i > k_ . The normalized value _x_ _[′]_ has bitwidth _ℓ_ and scale _ℓ_ _−_ 2.
Next, we use _g_ most significant bits of _z_ _[′]_, i.e., _e_ and the parity
of _k −_ _s_, i.e., _B_, to compute the initial approximation via a
lookup table _L_ rsqrt whose entries are as follows:



_D. Formal verification of our Math functionalities_


It is desirable for math libraries to have a formal proof of
correctness about their purported numerical precision. Such
a proof establishes that for all possible inputs, the ULP
error (Section III) between the math implementation and the
exact real result is small. For small bitwidths (e.g. ⩽ 32)
that are used in ML (Section VI-C), it is tractable to prove
these bounds on ULP error using _exhaustive testing_, whereas
for 64-bit floating-point or 64-bit fixed-point math libraries,
these proofs can either be interactive [54], [77] or fully
automatic [38], [78], [108]. Since our focus is on math libraries
for ML, we choose the exhaustive testing approach for our
math library, specifically, we 1) run our implementations on
all possible inputs, 2) compare the ULP error between each
output and the infinite precision real result, and 3) report the
maximum observed ULP error as the bound. For step 2, we
need the ability to compute math functions to arbitrary degrees
of precision – this is offered by the GNU MPFR library [45].
We prove ULP error bounds for bitwidth 16 (Section VI-C)
and appropriate input/output scales, _s_ _x_ and _s_ _y_, and choose
parameters _d_, _g_, and _t_ accordingly to ensure high precision.
Note that given a bitwidth _ℓ_, a proof via exhaustive testing
requires 2 _[ℓ]_ tests. For exponential, we set _d_ = 8 and prove that
_∀s_ _x_ _, s_ _y_ _∈_ [8 _,_ 14], the maximum ULP error is 3. For sigmoid
and tanh, we set _d_ = 8, _g_ = _⌈_ _[s]_ _[y]_ 2 _[−]_ [2] _⌉_ and _t_ = 0, and prove that

_∀s_ _x_ _, s_ _y_ _∈_ [8 _,_ 14] the maximum ULP error is 3 for sigmoid
and 4 for tanh. For reciprocal square root, we choose inputs
_x_ ⩾ _ϵ_ where _ϵ_ = 0 _._ 1, and set _g_ = _⌈_ _[s]_ 2 _[y]_ _[⌉]_ [and] _[ t]_ [ = 1][. We prove]

that _∀s_ _x_ _, s_ _y_ _∈_ [4 _,_ 13], the maximum ULP error is 4.
Thus, using exhaustive testing, we prove that our math
implementations are precise for chosen parameters and provide
standard precision guarantees that are expected from math
libraries viz. ULP error _<_ 5; Intel’s SVML [4] also provides
math implementations with 4 ULP error. We use the same
parameter setting described above for the empirical evaluation.


VI. E VALUATION


In this section, we empirically compare our protocols for
math functions with prior works and describe the results of our
ML case studies. The closest work to ours is MiniONN [83],
the only prior work on secure inference that has been evaluated
on an RNN. MiniONN proposes a recipe to obtain piecewise
linear approximations to sigmoid/tanh that are then evaluated
using its protocols. Our secure implementations of sigmoid are
an order of magnitude better in communication (Table I). Note
that no prior work on 2-party secure inference (including MiniONN) provides secure implementations of exponentiation and
reciprocal square root; we evaluate them in Table II. Generalpurpose MPC frameworks like MP-SPDZ [66] also provide
semi-honest 2PC implementations of math functions [3] that
are compatible with the standard (power-of-2 ring-based)
fixed-point representation. However, the communication of our
protocols is up to two orders of magnitude lower. Alternatives
that use representations such as field-based representations or
floating-point also suffer from high communication overheads.



�



_L_ rsqrt ( _e||B_ ) = Fix



1
� ~~�~~ ( _B_ + 1)(1 + urt ( _g,g_ ) ( _e_ )) _[, g]_ [ + 4] _[, g]_ [ + 2]



1
� ~~�~~ ( _B_ + 1)(1 +



**Functionality** _F_ rsqrt _[ℓ,s,ℓ,s]_ _[′]_ ( _⟨x⟩_ _[ℓ]_ )
Normalizes _x_ to _x_ _[′]_ as follows:



1) _k_ = MSNZB( _x_ ) _∈_ [ _ℓ_ ].
2) _A_ = 2 _[ℓ][−]_ [2] _[−][k]_, _B_ = ( _s −_ _k_ ) mod 2.
3) _C_ = 2 _[⌈]_ _[s][−]_ 2 _[k]_ _⌉_ + _⌊_ _[ℓ][−][s]_ 2 _[−]_ [1] _⌋_ .



2 _[k]_ _⌉_ + _⌊_ _[ℓ][−][s]_ 2 _[−]_ [1]



3) _C_ = 2 _[⌈]_ _[−]_ 2 _⌉_ + _⌊_ _[−]_ 2 _[−]_ _⌋_ .

4) _x_ _[′]_ = _x ∗_ _ℓ_ _A_ .

Computes the initial approximation _w_ as follows:



1) _x_ _[′]_ = _d||e||f, d ∈{_ 0 _,_ 1 _}_ [2] _, e ∈{_ 0 _,_ 1 _}_ _[g]_ _, f ∈{_ 0 _,_ 1 _}_ _[ℓ][−]_ [2] _[−][g]_ .
2) _w_ = _L_ rsqrt ( _e||B_ ), _w ∈{_ 0 _,_ 1 _}_ _[g]_ [+4] .

Goldschmidt’s method for _t_ iterations:

1) _x_ _[′′]_ = TR( _x_ _[′]_ _, ℓ_ _−_ 3 _−_ _s_ _[′]_ ) _, q_ 0 = _B_ ? _x_ _[′′]_ : TR( _x_ _[′]_ _,_ 1).
2) _a_ 0 = 2 _[s]_ _[′]_ _[−][g][−]_ [2] _∗_ _s_ _′_ +2 _w, p_ 0 = _a_ 0 .
3) For _i ∈{_ 1 _, . . ., t}_ do

a) _Y_ _i_ = TR( _p_ _i−_ 1 _∗_ 2 _s_ _′_ +2 _p_ _i−_ 1 _, s_ _[′]_ ).
b) _q_ _i_ = TR( _q_ _i−_ 1 _∗_ 2 _s_ _′_ +2 _Y_ _i_ _, s_ _[′]_ ).

c) _p_ _i_ = 3 _·_ 2 _[s]_ _[′]_ _[−]_ [1] _−_ ( _q_ _i_ _≫_ _A_ 1).
d) _a_ _i_ = TR( _a_ _i−_ 1 _∗_ 2 _s_ _′_ +2 _p_ _i_ _, s_ _[′]_ ).
Uses reciprocal square root of _x_ _[′]_ to compute the same for _x_ :

_ℓ−s−_ 1
1) Return TR( _a_ _t_ _∗_ _ℓ/_ 2+ _s_ _′_ +3 _C,_ � 2 �) mod _L_ .


Fig. 6: _The functionality F_ rsqrt _[ℓ,s,s]_ _[′]_ _[for parameters][ g, t][.]_


We formally verify that for _ℓ, s, s_ _[′]_ in our evaluation, our
choice of _g, t_ ensures precise results for _F_ rsqrt (Section V-D).
We build a protocol for _F_ rsqrt as follows: We consider
the functionality _F_ MSNZB that outputs the shares of one-hot
encoding of MSNZB( _x_ ) and give a protocol for the same
in Appendix F. It is easy to compute the terms _A, B, C_
using dot-products of this one-hot vector with publicly known
vectors. For our initial approximation, we rely on protocols
for _F_ DigDec and _F_ LUT . The Goldschmidt’s iterations crucially
utilize arithmetic over variable bitwidth and truncation operations and each of these steps require our efficient protocols
from Section IV. Table II gives our concrete numbers and
compares with prior work.


10 Although we would have liked to use the initial approximation provided
by [63], there seems to be some typographical errors in the published
equations and we are unable to correct them.


Next, we evaluate our library S I R NN for DNN inference on
end-to-end ML models. First, we evaluate S I R NN on models
with math functions considered by priors works [83], [102].
Since they evaluate sigmoid and tanh using generic 2PC protocols, S I R NN has an order of magnitude less communication
(Table III). Next, we evaluate S I R NN on RNNs for sports
training and audio keyword spotting that use GRU cells, which
are composed of sigmoid and tanh operators. There are two
ways to securely evaluate our math functionalities, with our
2PC protocols and with generic 2PC protocols for mixed
arithmetic and boolean compute [25], [32], [41], [95]. We
evaluate both and observe that S I R NN communicates over

500 _×_ less data for both the RNNs (Table IV). Finally, we
evaluate S I R NN on a recent model architecture that combines

CNN operators and RNN operators to find the human heads
in images with state-of-the-art accuracy [104]. We provide
the first secure implementation for this complex model; its
secure implementation requires all the protocols described in
this paper including reciprocal square root and takes less than
7 minutes on our evaluation set up:
_System Details._ We use a set up where the 2 machines are
connected via a 377 MBps LAN network with 0.8 ms RTT.
Both the machines have commodity hardware with a 4-core
3.7 GHz Xeon processor and 16 GBs of RAM.
_Implementation Details._ The users of S I R NN express their
DNNs as a combination of calls to S I R NN ’s C++ library
functions. These functions include matrix multiplication, convolutions, MBConv blocks, L2 Normalization, batch normalization, broadcasting; pointwise operators like sigmoid, tanh,
exponential, reciprocal square root, matrix addition, Hadamard
product; comparison-based operators like argmax, maxpool,
ReLU, and ReLU6. The last four functions use protocols
from [99] and the rest use our building blocks. The library
functions take scales as arguments and are templated on
the bitwidths. The S I R NN library is implemented using 28K
lines of C++. We statically generate 36 LUTs that consume
additional 35K LOC.


_A. Microbenchmarks_


_a) Sigmoid:_ In Table I, we compare our protocol with
prior work for generating sigmoid output with 12-bits of
precision (i.e., scale 12). We report absolute numbers for
time taken and communication for both our protocols and
prior work, as well as improvement factor of our protocols
in parentheses. We follow this pattern for all the tables
in this section. We focus on sigmoid as the numbers for
tanh are similar. One sigmoid evaluation with our protocols
incurs less than 5KB of communication and produces precise
results with at most 3 ULPs error. In ML, sigmoid is usually
computed pointwise over all the entries in a tensor. Hence,
one needs to compute sigmoid of a large number of instances
when dealing with realistic ML benchmarks. Although the
communication to compute _n_ sigmoid instances grows linearly
with _n_, empirically we have observed that the time taken or the
latency grows sub-linearly with _n_ (columns 2 to 5 of Table I),
which helps our implementations to scale well to large tensors



(Section VI-C). The cost of rounds amortizes better for large
tensors resulting in the sub-linear growth in latency.
As a baseline, we consider the recipe of MiniONN that
approximates math functions with piecewise linear approximations and provides protocols to evaluate these splines. More
precise approximations require more number of pieces. To get
an ULP error below 5, MiniONN needs a 48-way spline which
provides poor performance when evaluated securely because
of a 70 _×_ communication overhead.

For the RNN benchmark that MiniONN considers (Section VI-B), the precision offered by the 48-piece spline is an
overkill and a 12-piece spline suffices to maintain the cross
entropy loss. Although this 12-piece spline is more efficient
than 48-piece spline, its performance is still much worse than
our protocols and incurs a 19 _×_ communication overhead.
Furthermore, this 12-piece spline incurs an error of 104 ULPs.
Hence, our implementations are superior in both precision and
performance. While a 12-piece spline suffices for this benchmark, MiniONN remarks that other benchmarks need splines
with more number of pieces that are even more expensive to
compute. Because our implementations are guaranteed to be
numerically precise, they can be used as-is with no loss in
model accuracy (Section VI-C).
DeepSecure [102] uses garbled circuits (GC) to evaluate
DNNs that use sigmoid and tanh activations. We checked with
the authors of DeepSecure and the circuits for math functions
are not available. Hence, we cannot compute the ULP errors
of their implementations. However, DeepSecure reports the
number of non-XOR gates that can be used for performance
estimates. We used state-of-the-art for GC implementation,
i.e., EMP-Toolkit [1], [52], [53], to obtain these performance
estimates that are better than the performance reported by
DeepSecure. The communication of our protocols is 25 _×_
lower (4 [th] row of Table I).
MP-SPDZ [66], a general-purpose MPC framework, provides 2 baseline sigmoid implementations for 2PC [3]: Polybased, which uses a range reduction and Taylor series polynomials to compute exponential followed by division, and
PL-based, which is a built-in piecewise linear spline. The
former implementation incurs error comparable to us but
communicates 201 _×_ more, while the latter is more than an
order of magnitude inferior in precision and communication
(5 [th] and 6 [th] row of Table I).
While we focus on power-of-2 rings, there are other works
on secure implementations of sigmoid that use field-based or
floating point representations. Field-based protocols perform
poorly for non-linear computations like truncation and comparisons, which are abundant in fixed-point representations
of DNNs [64], [90], [99]. Similarly, it is well-known that
the protocols over floating-point are much slower than fixedpoint [29], [73]. Nonetheless, for completeness, we compare
against the state-of-the-art field-based implementations in MPSPDZ [3], [66] and they perform worse (7 [th] and 8 [th] rows of
Table I). We also compare with floating-point implementations
of math functions provided by ABY [40] and EMP-Toolkit [1];
our protocols are at least 90 _×_ better in communication per


|Inference<br>Benchmark|Runtime (in sec)|Col3|Comm.|Col5|
|---|---|---|---|---|
|Inference<br>Benchmark|Prior|Our Work|Prior|Our Work|
|MiniONN LSTM|1_._1<br>(2_._2x)|0_._48|182 MB<br>(19_._5x)|9_._32 MB|
|DeepSecure B4|465<br>(87x)|5_._3|83_._7 GB<br>(43x)|1_._94 GB|









TABLE III: Comparison with benchmarks from MiniONN [83] and DeepSecure [102].


_B. Prior DNNs_


In Table III, we evaluate our protocols on benchmarks with
math functions from MiniONN [83] and DeepSecure [102].
MiniONN evaluated an LSTM for text data which has 2

LSTM layers each with 800 instances of sigmoid and 200
instances of tanh. Our protocols incur an order of magnitude
less communication for these instances. We consider the

largest benchmark of DeepSecure, B4, with 2 tanh layers of
2000 and 500 instances, which classifies sensor data into 19
different physical activities. To estimate the time taken by
DeepSecure on our setup, we ran a circuit with the same nonXOR complexity as B4 using EMP-Toolkit [1] (similar to our
microbenchmarks) that provides better performance than the
communication and latency in [102]. Our protocols have 87 _×_
lower latency and 43 _×_ lower communication.


_C. Case studies_


We demonstrate the applicability of secure inference to three
new domains that no prior work has considered before: RNNs
applied to time series sensor data, RNNs applied to speech
data, and combining CNNs and RNNs to identify human heads
in images. The feasibility of our case studies crucially relies on
our efficient protocols for math functions. Our first case study
is an industrial model (Industrial [72]) which uses an RNN
with GRU cells to provide feedback on the quality of shots
in a bat-and-ball game from the data obtained from sensors
deployed on the bat. Second, we evaluate an RNN (Google30 [74]) for keyword spotting in the standard Google-30 [112]
dataset that identifies simple commands, digits, and directions
from speech data obtained from thousands of people. Third,
the head detection model (Heads [104]) combines CNNs and
RNNs for the best accuracy on the SCUT Head dataset [96].
It uses inverted residual blocks, or MBConv blocks [105],
for efficient convolutions. Instead of simple pooling operators
like maxpool or average pool, it uses RNN-based pooling
that provides high accuracy. We summarize the input fixedpoint code of these benchmarks below. These fixed-point C++
programs were automatically generated from high-level ML
models by [72] (a compiler for embedded devices) and linked
with S I R NN . All of the benchmarks use a mixture of variables

with bitwidth 8, 16, and 32 with 16 being the bitwidth used
for input and output of the math functions.


_•_ _Industrial-72_ : It contains 7 sigmoid and 7 tanh layers,
with 64 instances each. While sigmoid uses the input
scale 8 and output scale 14, for tanh both scales are 8.


|Technique|Total Time for #Instances (in sec)|Col3|Col4|Col5|Comm./<br>Instance<br>(in KB)|Max<br>ULP<br>Err.|
|---|---|---|---|---|---|---|
|Technique|102|103|104|105|105|105|
|Our Work|0_._08|0_._10|0_._25|1_._58|4_._88|3|
|MiniONN<br>48-piece|0_._20<br>(2_._5x)|1_._94<br>(19_._4x)|18_._85<br>(75x)|182_._2<br>(115x)|341_._03<br>(70x)|4|
|MiniONN<br>12-piece|0_._06<br>(0_._8x)|0_._54<br>(5_._4x)|5_._24<br>(21x)|53_._84<br>(34x)|93_._36<br>(19_._1x)|104|
|Deep-<br>Secure|0_._16<br>(2x)|0_._84<br>(8_._4x)|8_._1<br>(32x)|141_._3<br>(89x)|124_._65<br>(25x)|NA|
|MP-SPDZ<br>Ring Poly|0_._75<br>(9_._4x)|1_._72<br>(17_._2x)|14_._88<br>(59_._5x)|140_._6<br>(89x)|981_._11<br>(201x)|2|
|MP-SPDZ<br>Ring PL|0_._27<br>(3_._4x)|0_._28<br>(2_._8x)|1_._32<br>(5_._3x)|12_._34<br>(7_._8x)|76_._42<br>(15_._7x)|266|
|MP-SPDZ<br>Field Poly|0_._91<br>(11_._4x)|1_._91<br>(19_._1x)|16_._51<br>(66x)|127<br>(80x)|228_._63<br>(46_._9x)|2|
|MP-SPDZ<br>Field PL|0_._52<br>(6_._5x)|0_._47<br>(4_._7x)|1_._79<br>(7_._2x)|14_._23<br>(9x)|27_._52<br>(5_._6x)|266|



TABLE I: Comparison with prior works on sigmoid with
varying number of instances.






























|Technique|Total Time for #Instances (in sec)|Col3|Col4|Col5|Comm./<br>Instance<br>(in KB)|Max<br>ULP<br>Error|
|---|---|---|---|---|---|---|
|Technique|102|103|104|105|105|105|
|Exponentiation|Exponentiation|Exponentiation|Exponentiation|Exponentiation|Exponentiation|Exponentiation|
|Our Work|0.03|0.04|0.15|1.00|2.12|3|
|MP-SPDZ|0_._34<br>(11_._3x)|0_._56<br>(14x)|3_._90<br>(26x)|35_._95<br>(35_._9x)|254_._95<br>(120x)|2|
|Reciprocal Square Root|Reciprocal Square Root|Reciprocal Square Root|Reciprocal Square Root|Reciprocal Square Root|Reciprocal Square Root|Reciprocal Square Root|
|Our Work|0_._13|0_._13|0_._30|1_._84|6|4|
|MP-SPDZ|0_._94<br>(7_._2x)|3_._90<br>(30x)|35_._87<br>(120x)|338_._9<br>(184x)|2535<br>(423x)|8|



TABLE II: Comparison with (power-of-2) ring-based MPSPDZ protocols with varying number of instances.


instance and 97 _×_ better in runtime (for 10 [5] instances).


Finally, SecureML [92] and ABY2.0 [95] use a 3-piece linear spline to approximate sigmoid. This simple implementation
has a whopping error of 1547 ULPs and tanks the accuracy of
our RNN benchmarks. For instance, it leads to a tremendous
drop in accuracy of the Google-30 network from 84.4% (with
our sigmoid implementation) to 60.95%. The insufficiency
of this approximation has also been noted by [83] where it
caused the cross-entropy loss to diverge to infinity. Hence, this
crude approximation is usable only in restricted contexts and
is unsuitable for generic math libraries, which is our aim here.


_b) Exponential and reciprocal square-root:_ Table II
shows the comparison of our exponentiation and reciprocal
square-root protocols with power-of-2 ring based protocols in
MP-SPDZ framework (for scale 12). It has native support for
exponentiation. We implement reciprocal square root in MPSPDZ by calling its built-in functions for square root and
reciprocal. As the table shows, our protocols are orders of
magnitude better, both in terms of time-taken and communication, and provide better or comparable ULP errors.


the-art 2PC protocols that have been used by recent work
on secure inference [25], [32], [83], [92]. We have added a
new code generator to [72] that generates E Z PC [32] code
which is then automatically translated to ABY code. Other
generic protocols that have suitable frontends [60], [82], [88],

[109], like garbled circuits, are several orders of magnitude
slower than ABY [32], [92]: ML inference involves many
multiplications that are very expensive with garbled circuits.
S I R NN is over 500 _×_ better than ABY in communication and

more than an order of magnitude faster in runtime. Without
our protocols, it takes almost an hour to run Google-30. This
situation is further exacerbated on bigger models and running
the Heads model with ABY is intractable because it requires
hundreds of terabytes of communication. With batching, the
performance differences are stark: S I R NN is three orders of
magnitude better in latency and communication compared to
the ABY baseline.


VII. O THER R ELATED WORK


Prior 2PC works that use high degree polynomials for
approximating math functions [9], [34], [57], [68] need degree
7 or higher to maintain accuracy. In the course of this
work, we have observed that evaluating polynomials with
degree 3 or higher with 2PC is much more expensive than
the LUT-based implementations of Section V. Some prior
works on secure inference implement math functions with
ad hoc approximations that can lose model accuracy: e.g.
SecureML [92] and ABY2.0 [95] use a crude 3-piece linear
approximation, Ball et al. [13] replace tanh with the signum
function, and Glyph [85] and Nandakumar et al. [93] use
tables of approximate results Most recent works on secure
inference limit their evaluation to benchmarks that don’t use

math functions [17], [39], [44], [47], [64], [90], [99]. Prior
2PC works that use floating-point representations (instead of
fixed-point representations) have much higher performance
overheads [1], [6], [7], [12], [33], [40], [46], [66], [84]
Other relevant works that need additional parties to ensure security such as 3PC with honest majority or 2PC
with trusted dealer include [8], [9], [28]–[31], [43], [86],
Chameleon [101], CrypTen [69], TF-Encrypted [2], CrypTFlow [73], PySyft [103], ABY [3] [91], SecureQ8 [37], and
Sharemind [65], [67], [71], [75], [97]. Some of these works
have considered approximations to math functions and, similar
to 2PC works, they either use polynomial-based approximations (e.g. [9], [71], [86]) or work over floating-point (e.g. [8],

[29], [30], [65], [67], [75], [97]). Kerik et al. [67] also consider
building blocks such as extension, truncate-and-reduce, and
multiplication of non-uniform bitwidths in the 3PC context.
In terms of representations, while floating-point and fixedpoint representations are most common, [43] proposed the new
representations of golden-section and logarithmic numbers and
evaluated using 3PC protocols.
Recent works on silent-OT [22], [114] provide OT extensions with much lower communication than IKNP-style extensions [62], at the cost of higher computational overhead. Since



|Benchmark|Batch|Runtime (sec)|Col4|Comm.|Col6|
|---|---|---|---|---|---|
|Benchmark|Batch|[41]|SIRNN|[41]|SIRNN|
|Industrial-72|1|68_._33<br>(18x)<br>|3_._7|11_._84 GB<br>(510x)<br>|23_._8 MB|
|Industrial-72|128|8746~~_∗_~~<br>(661x)|13_._2|1_._47 TB~~_∗_~~<br>(1451x)|1_._04 GB|
|Google-30|1|3337<br>(67x)<br>|49_._6|259 GB<br>(574x)<br>|0_._45 GB|
|Google-30|128|4_._3x10~~5~~~~_∗_~~<br>(3050x)|140|32_._38 TB~~_∗_~~<br>(1316x)|25_._2 GB|
|Heads|1|NA|409_._7|NA|85_._5 GB|


*extrapolated, the run could not be completed due to TB comm.


TABLE IV: Secure inference on DNNs using S I R NN and [41].


_•_ _Google-30_ : It contains 99 sigmoid and 99 tanh layers,
with 100 instances each. While sigmoid uses the input
scale 6 and output scale 14, for tanh both scales are 6.

_•_ _Heads_ : It contains 128 sigmoid and 128 tanh layers, with
18096 instances each. While sigmoid uses the input scale
11 and output scale 14, for tanh both scales are 11.
Additionally, the benchmark contains 8 sigmoid and 8
tanh layers, with 72384 instances each. For these layers,
sigmoid uses the input scale 13 and output scale 14, and
for tanh both scales are 13. Finally, it also contains 3 L2Normalise layers that have 1200, 1200 and 300 reciprocal
square-root operations. The layers have input scales 12,
10 and 12 and output scales 11, 9 and 11, respectively.


Note that the Heads model makes about 3 million calls to

sigmoid/tanh, which is three orders of magnitude larger than
the number of calls to these functions in the benchmarks used

by prior work (Section VI-B).
In Table IV, we present the latency and communication
required by S I R NN on above benchmarks. Using our protocols,
Industrial takes 4 seconds, Google-30 takes under a minute,
and Heads takes less than 7 minutes. The time per inference
can be further improved by _batching_ multiple predictions. For
a batch size of 128, the amortized time per inference of Industrial is 0.1s and of Google-30 is 1.1s! The savings in batching
come from amortizing the networking cost by packing data
from multiple inference queries. Owing to the high numerical
precision of our math functionalities (Section V-D), S I R NN
either matches or exceeds the model accuracy of the provided
fixed-point ML model. In Heads, about half the time is spent
in math operations and the rest of the time is spent in matrix
multiplications, convolutions, and Hadamard products. The
good performance on end-to-end benchmarks is a result of codesigning precise math functionalities and efficient protocols.
Next, we perform an ablation study. In particular, the fixedpoint code with our math functionalities can be run with
other protocols. However, prior work on secure inference don’t
support juggling between different bitwidths that our math
functionalities require. Hence, for running these functionalities
with any prior protocol, we need to use an appropriately
large uniform bitwidth. We evaluate our benchmarks with
ABY [41] using the necessary bitwidth of 64 as a baseline
in Table IV. ABY [41] provides general purpose state-of

our protocols make use of OTs in a black-box manner, silentOT can be used to obtain lower communication. However,
in our setting, when the IKNP-OT instances are computed
by multiple threads and are “load-balanced” (i.e., each party
plays the role of the sender in half the OT instances and as the
receiver in the other half), we empirically observe that IKNPstyle extensions are more performant than silent-OT in our
LAN evaluation environment. Hence, S I R NN uses IKNP-style
OT extensions in Section VI.


VIII. C ONCLUSION


We presented novel secure implementations of math functions that rely on cryptographic protocols for mixed-bitwidths.
These implementations, with up to 423 _×_ lower communication than the state-of-the-art, help us evaluate ML models that
have three orders of magnitude more calls to math functions
than benchmarks considered by prior work. Compared to a
baseline, S I R NN achieves three orders of magnitude lower
communication and latency. While prior work on secure 2party inference has focused on image analysis, S I R NN provides the first implementations of RNNs operating on speech
data, sensor data, and, in combinations with CNNs, detecting
heads with state-of-the-art accuracy. Because of high numerical precision of our math implementations, there is no loss in
model accuracy over cleartext. Although, in this work, we have
focused on particular functions that occur in many ML models,
the recipe of look ups followed by Newton Raphson iterations
to obtain precise functionalities is well-known in embedded
systems and can be instantiated for other math functions as
well. We believe that our novel 2PC protocols would help
provide the building blocks necessary for such functionalities.


A CKNOWLEDGEMENT


We thank Pratik Bhatu, Aayan Kumar, and Aditya Kusupathi
for their help with the implementation and the evaluation.


R EFERENCES


[[1] “EMP-toolkit: Efficient MultiParty computation toolkit,” https://github.](https://github.com/emp-toolkit)
[com/emp-toolkit, 2016.](https://github.com/emp-toolkit)

[2] “TF-Encrypted: A Framework for Encrypted Machine Learning in
[TensorFlow,” https://github.com/tf-encrypted/tf-encrypted, 2018.](https://github.com/tf-encrypted/tf-encrypted)

[3] “Multi-Protocol SPDZ: Versatile framework for multi-party computa[tion,” 2019. [Online]. Available: https://github.com/data61/MP-SPDZ](https://github.com/data61/MP-SPDZ)

[[4] “Intel SVML,” https://software.intel.com/content/www/us/en/develop/](https://software.intel.com/content/www/us/en/develop/documentation/mkl-vmperfdata/top.html)
[documentation/mkl-vmperfdata/top.html, 2020.](https://software.intel.com/content/www/us/en/develop/documentation/mkl-vmperfdata/top.html)

[5] N. Agrawal, A. S. Shamsabadi, M. J. Kusner, and A. Gasc´on, “QUOTIENT: Two-Party Secure Neural Network Training and Prediction,”
in _CCS 2019_ .

[6] M. Aliasgari and M. Blanton, “Secure computation of hidden markov
models,” in _SECRYPT_, 2013.

[7] M. Aliasgari, M. Blanton, and F. Bayatbabolghani, “Secure computation of hidden markov models and secure floating-point arithmetic in
the malicious model,” _Int. J. Inf. Sec._, 2017.

[8] M. Aliasgari, M. Blanton, Y. Zhang, and A. Steele, “Secure computation on floating point numbers,” in _NDSS_, 2013.

[9] A. Aly and N. P. Smart, “Benchmarking privacy preserving scientific
operations,” in _ACNS_, 2019.

[10] D. W. Archer, J. M. Calder´on Trilla, J. Dagit, A. Malozemoff,
Y. Polyakov, K. Rohloff, and G. Ryan, “Ramparts: A programmerfriendly system for building homomorphic encryption applications,” in
_WAHC 2019_ .




[11] G. Asharov, Y. Lindell, T. Schneider, and M. Zohner, “More efficient
oblivious transfer and extensions for faster secure computation,” in _CCS_
_2013_ .

[12] S. Bai, G. Yang, J. Shi, G. Liu, and Z. Min, “Privacy-preserving
oriented floating-point number fully homomorphic encryption scheme,”
_Secur. Commun. Networks 2018_ .

[13] M. Ball, B. Carmer, T. Malkin, M. Rosulek, and N. Schimanski,
“Garbled Neural Networks are Practical,” _ePrint 2019/338_ .

[14] P. Banerjee, D. Bagchi, M. Haldar, A. Nayak, V. Kim, and R. Uribe,
“Automatic conversion of floating point matlab programs into fixed
point fpga based hardware design,” in _FCCM 2003_ .

[15] D. Beaver, “Efficient Multiparty Protocols Using Circuit Randomization,” in _CRYPTO 1991_ .

[16] G. R. Blakley, “Safeguarding cryptographic keys,” in _Managing Re-_
_quirements Knowledge, International Workshop on_, 1979.

[17] F. Boemer, R. Cammarota, D. Demmler, T. Schneider, and H. Yalame,
“MP2ML: a mixed-protocol machine learning framework for private
inference,” in _ARES 2020_ .

[18] F. Boemer, A. Costache, R. Cammarota, and C. Wierzynski, “nGraphHE2: A High-Throughput Framework for Neural Network Inference
on Encrypted Data,” in _WAHC 2019_ .

[19] F. Boemer, Y. Lao, R. Cammarota, and C. Wierzynski, “nGraph-HE:
A Graph Compiler for Deep Learning on Homomorphically Encrypted
Data,” in _CF 2019_ .

[20] C. Boura, N. Gama, and M. Georgieva, “Chimera: a unified framework
for B/FV, TFHE and HEAAN fully homomorphic encryption and
predictions for deep learning,” _ePrint 2018/758_ .

[21] E. Boyle, N. Chandran, N. Gilboa, D. Gupta, Y. Ishai, N. Kumar, and
M. Rathee, “Function Secret Sharing for Mixed-Mode and Fixed-Point
Secure Computation,” _ePrint 2020/1392_ .

[22] E. Boyle, G. Couteau, N. Gilboa, Y. Ishai, L. Kohl, P. Rindal, and
P. Scholl, “Efficient two-round OT extension and silent non-interactive
secure computation,” in _CCS_ . ACM, 2019, pp. 291–308.

[23] L. Braun, D. Demmler, T. Schneider, and O. Tkachenko, “MOTION

   - A Framework for Mixed-Protocol Multi-Party Computation,” _ePrint_
_2020/1137_ .

[24] D. Brooks and M. Martonosi, “Dynamically exploiting narrow width
operands to improve processor power and performance,” in _HPCA_
_1999_ .

[25] N. B¨uscher, D. Demmler, S. Katzenbeisser, D. Kretzmer, and T. Schneider, “HyCC: Compilation of Hybrid Protocols for Practical Secure
Computation,” in _CCS 2018_ .

[26] R. Canetti, “Security and Composition of Multiparty Cryptographic
Protocols,” _J. Cryptology 2000_ .

[27] S. Carpov, P. Dubrulle, and R. Sirdey, “Armadillo: A compilation chain
for privacy preserving applications,” in _SCC 2015_ .

[28] O. Catrina, “Round-efficient protocols for secure multiparty fixed-point
arithmetic,” in _COMM 2018_ .

[29] ——, “Efficient Secure Floating-point Arithmetic using Shamir Secret
Sharing,” in _ICETE (2)_, 2019.

[30] ——, “Evaluation of floating-point arithmetic protocols based on
shamir secret sharing,” in _ICETE (Selected Papers)_, 2019.

[31] O. Catrina and A. Saxena, “Secure computation with fixed-point
numbers,” in _Financial Cryptography_, 2010.

[32] N. Chandran, D. Gupta, A. Rastogi, R. Sharma, and S. Tripathi,
“EzPC: Programmable and Efficient Secure Two-Party Computation
for Machine Learning,” in _IEEE EuroS&P 2019_ .

[33] Y. Chang and C. Lu, “Oblivious polynomial evaluation and oblivious
neural learning,” in _ASIACRYPT_, 2001.

[34] V. Chen, V. Pastro, and M. Raykova, “Secure Computation for Machine
Learning With SPDZ,” in _PPML 2018, NeurIPS 2018 Workshop_ .

[35] I. Chillotti, N. Gama, M. Georgieva, and M. Izabach`ene, “Faster fully
homomorphic encryption: Bootstrapping in less than 0.1 seconds,” in
_ASIACRYPT 2016_ .

[36] K. Cho, B. van Merrienboer, D. Bahdanau, and Y. Bengio, “On the
properties of neural machine translation: Encoder-decoder approaches,”
in _SSST-8, 2014_ .

[37] A. P. K. Dalskov, D. Escudero, and M. Keller, “Secure evaluation of
quantized neural networks,” _PoPETs 2020_ .

[38] E. Darulova and V. Kuncak, “Sound compilation of reals,” in _POPL_
_2014_ .

[39] R. Dathathri, O. Saarikivi, H. Chen, K. Lauter, S. Maleki, M. Musuvathi, and T. Mytkowicz, “CHET: An Optimizing Compiler for FullyHomomorphic Neural-Network Inferencing,” in _PLDI 2019_ .


[40] D. Demmler, G. Dessouky, F. Koushanfar, A. Sadeghi, T. Schneider,
and S. Zeitouni, “Automated synthesis of optimized circuits for secure
computation,” in _CCS 2015_ .

[41] D. Demmler, T. Schneider, and M. Zohner, “ABY - A Framework
for Efficient Mixed-Protocol Secure Two-Party Computation,” in _NDSS_
_2015_ .

[42] G. Dessouky, F. Koushanfar, A. Sadeghi, T. Schneider, S. Zeitouni, and
M. Zohner, “Pushing the Communication Barrier in Secure Computation using Lookup Tables,” in _NDSS 2017_ .

[43] V. Dimitrov, L. Kerik, T. Krips, J. Randmets, and J. Willemson,
“Alternative implementations of secure real numbers,” in _CCS 2016_ .

[44] D. Escudero, S. Ghosh, M. Keller, R. Rachuri, and P. Scholl, “Improved Primitives for MPC over Mixed Arithmetic-Binary Circuits,”
in _CRYPTO 2020_ .

[45] L. Fousse, G. Hanrot, V. Lef`evre, P. P´elissier, and P. Zimmermann,
“MPFR: A multiple-precision binary floating-point library with correct
rounding,” _ACM Trans. Math. Softw._, 2007.

[46] M. Franz and S. Katzenbeisser, “Processing encrypted floating point
signals,” in _MM&Sec 2011_ .

[47] R. Gilad-Bachrach, N. Dowlin, K. Laine, K. E. Lauter, M. Naehrig,
and J. Wernsing, “CryptoNets: Applying Neural Networks to Encrypted
Data with High Throughput and Accuracy,” in _ICML 2016_ .

[48] D. Goldberg, “What every computer scientist should know about
floating-point arithmetic,” _ACM Comput. Surv._, 1991.

[49] O. Goldreich, S. Micali, and A. Wigderson, “How to Play any Mental
Game or A Completeness Theorem for Protocols with Honest Majority,” in _ACM STOC 1987_ .

[50] R. E. Goldschmidt, “Applications of division by convergence,” M.S.
thesis, MIT, 1964.

[51] S. Gopinath, N. Ghanathe, V. Seshadri, and R. Sharma, “Compiling
KB-Sized Machine Learning Models to Tiny IoT Devices,” in _PLDI_
_2019_ .

[52] C. Guo, J. Katz, X. Wang, C. Weng, and Y. Yu, “Better concrete
security for half-gates garbling (in the multi-instance setting),” in
_CRYPTO (2)_, 2020.

[53] C. Guo, J. Katz, X. Wang, and Y. Yu, “Efficient and secure multiparty
computation from fixed-key block ciphers,” in _IEEE Symposium on_
_Security and Privacy_, 2020.

[54] J. Harrison, “A machine-checked theory of floating point arithmetic,”
in _Theorem Proving in Higher Order Logics_, 1999.

[55] T. Hastie, R. Tibshirani, and J. Friedman, _The Elements of Statistical_
_Learning (2nd Edition)_, 2009.

[56] K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for
Image Recognition,” in _CVPR 2016_ .

[57] B. Hemenway, S. Lu, R. Ostrovsky, and W. W. IV, “High-precision
secure computation of satellite collision probabilities,” in _SCN 2016_ .

[58] E. Hesamifard, H. Takabi, and M. Ghasemi, “CryptoDL: Deep Neural
Networks over Encrypted Data,” _CoRR 2017_ .

[59] S. Hochreiter and J. Schmidhuber, “Long Short-Term Memory,” _Neural_
_Computation_, 1997.

[60] A. Holzer, M. Franz, S. Katzenbeisser, and H. Veith, “Secure two-party
computations in ANSI C,” in _CCS 2012_ .

[61] G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger, “Densely
Connected Convolutional Networks,” in _CVPR 2017_ .

[62] Y. Ishai, J. Kilian, K. Nissim, and E. Petrank, “Extending Oblivious
Transfers Efficiently,” in _CRYPTO 2003_ .

[63] M. Ito, N. Takagi, and S. Yajima, “Efficient Initial Approximation
for Multiplicative Division and Square Root by a Multiplication with
Operand Modification,” _IEEE Transactions on Computers_, 1997.

[64] C. Juvekar, V. Vaikuntanathan, and A. Chandrakasan, “GAZELLE: A
Low Latency Framework for Secure Neural Network Inference,” in
_USENIX Security 2018_ .

[65] L. Kamm and J. Willemson, “Secure floating point arithmetic and
private satellite collision analysis,” _Int. J. Inf. Sec._, 2015.

[66] M. Keller, “MP-SPDZ: A versatile framework for multi-party computation,” in _CCS_, 2020.

[67] L. Kerik, P. Laud, and J. Randmets, “Optimizing MPC for robust and
scalable integer and floating-point arithmetic,” in _Financial Cryptogra-_
_phy Workshops 2016_ .

[68] D. Kim, Y. Son, D. Kim, A. Kim, S. Hong, and J. H. Cheon, “Privacypreserving approximate GWAS computation based on homomorphic
encryption,” _ePrint 2019/152_ .




[69] B. Knott, S. Venkataraman, A. Hannun, S. Sengupta, M. Ibrahim,
and L. van der Maaten, “CrypTen: Secure multi-party computation
meets machine learning,” in _Workshop on Privacy Preserving Machine_
_Learning, December 11, 2020_ .

[70] V. Kolesnikov and R. Kumaresan, “Improved OT Extension for Transferring Short Secrets,” in _CRYPTO 2013_ .

[71] T. Krips and J. Willemson, “Hybrid model of fixed and floating point
numbers in secure multiparty computations,” in _ISC_, 2014.

[72] A. Kumar, V. Seshadri, and R. Sharma, “Shiftry: RNN Inference in
2KB of RAM,” in _OOPSLA_, 2020.

[73] N. Kumar, M. Rathee, N. Chandran, D. Gupta, A. Rastogi, and
R. Sharma, “CrypTFlow: Secure TensorFlow Inference,” in _IEEE S&P_
_2020_ .

[74] A. Kusupati, M. Singh, K. Bhatia, A. Kumar, P. Jain, and M. Varma,
“FastGRNN: A Fast, Accurate, Stable and Tiny Kilobyte Sized Gated
Recurrent Neural Network,” in _NeurIPS 2018_ .

[75] P. Laud and J. Randmets, “A domain-specific language for low-level
secure multiparty computation protocols,” in _CCS 2015_ .

[76] S. Laur, H. Lipmaa, and T. Mielik¨ainen, “Cryptographically private
support vector machines,” in _SIGKDD 2006_ .

[77] J. Le Maire, N. Brunie, F. De Dinechin, and J. Muller, “Computing
floating-point logarithms with fixed-point operations,” in _IEEE ARITH_
_2016_ .

[78] W. Lee, R. Sharma, and A. Aiken, “On automatically proving the
correctness of math.h implementations,” in _POPL 2018_ .

[79] ——, “Verifying bit-manipulations of floating-point,” in _PLDI 2016_ .

[80] K.-P. Lin and M.-S. Chen, “Privacy-preserving outsourcing support
vector machines with random transformation,” in _SIGKDD 2010_ .

[81] Y. Lindell, _How to Simulate It – A Tutorial on the Simulation Proof_
_Technique_, 2017.

[82] C. Liu, X. S. Wang, K. Nayak, Y. Huang, and E. Shi, “ObliVM:
A Programming Framework for Secure Computation,” in _IEEE S&P_
_2015_ .

[83] J. Liu, M. Juuti, Y. Lu, and N. Asokan, “Oblivious Neural Network
Predictions via MiniONN Transformations,” in _CCS 2017_ .

[84] Y. Liu, Y. Chiang, T. Hsu, C. Liau, and D. Wang, “Floating point
arithmetic protocols for constructing secure data analysis application,”
in _KES_, 2013.

[85] Q. Lou, B. Feng, G. Charles Fox, and L. Jiang, “Glyph: Fast and
accurately training deep neural networks on encrypted data,” _to appear_
_in NeurIPS 2020_ .

[86] W.-j. Lu, Y. Fang, Z. Huang, C. Hong, C. Chen, H. Qu, Y. Zhou, and
K. Ren, “Faster secure multiparty computation of adaptive gradient
descent,” _to appear in PPML 2020, NeurIPS 2020 Workshop_ .

[87] E. Makri, D. Rotaru, N. P. Smart, and F. Vercauteren, “EPIC: Efficient
Private Image Classification (or: Learning from the Masters),” in _CT-_
_RSA 2019_ .

[88] D. Malkhi, N. Nisan, B. Pinkas, and Y. Sella, “Fairplay - Secure TwoParty Computation System,” in _USENIX Security 2004_ .

[89] D. Menard, D. Chillet, F. Charot, and O. Sentieys, “Automatic floatingpoint to fixed-point conversion for dsp code generation,” in _CASES_
_2002_ .

[90] P. Mishra, R. Lehmkuhl, A. Srinivasan, W. Zheng, and R. A. Popa,
“Delphi: A Cryptographic Inference Service for Neural Networks,” in
_USENIX Security 2020_ .

[91] P. Mohassel and P. Rindal, “ABY [3] : A Mixed Protocol Framework for
Machine Learning,” in _CCS 2018_ .

[92] P. Mohassel and Y. Zhang, “SecureML: A System for Scalable PrivacyPreserving Machine Learning,” in _IEEE S&P 2017_ .

[93] K. Nandakumar, N. K. Ratha, S. Pankanti, and S. Halevi, “Towards
deep neural network training on encrypted data,” in _CVPR Workshops_,
2019.

[94] A. Nayak, M. Haldar, A. Choudhary, and P. Banerjee, “Precision
and error analysis of matlab applications during automated hardware
synthesis for FPGAs,” in _DATE 2001_ .

[95] A. Patra, T. Schneider, A. Suresh, and H. Yalame, “ABY2.0: Improved
Mixed-Protocol Secure Two-Party Computation,” _to appear in USENIX_
_Security 2021_ .

[96] D. Peng, Z. Sun, Z. Chen, Z. Cai, L. Xie, and L. Jin, “Detecting heads
using feature refine net and cascaded multi-scale architecture,” _arXiv_
_2018_ .

[97] P. Pullonen and S. Siim, “Combining secret sharing and garbled
circuits for efficient private IEEE 754 floating-point computations,” in
_Financial Cryptography Workshops_, 2015.


[98] Y. Rahulamathavan, R. C. . Phan, S. Veluru, K. Cumanan, and
M. Rajarajan, “Privacy-preserving multi-class support vector machine
for outsourcing the data classification in cloud,” _TDSC 2014_ .

[99] D. Rathee, M. Rathee, N. Kumar, N. Chandran, D. Gupta, A. Rastogi,
and R. Sharma, “CrypTFlow2: Practical 2-Party Secure Inference,” in
_CCS 2020_ .

[100] M. S. Riazi, M. Samragh, H. Chen, K. Laine, K. E. Lauter, and
F. Koushanfar, “XONN: XNOR-based Oblivious Deep Neural Network
Inference,” in _USENIX Security 2019_ .

[101] M. S. Riazi, C. Weinert, O. Tkachenko, E. M. Songhori, T. Schneider, and F. Koushanfar, “Chameleon: A Hybrid Secure Computation
Framework for Machine Learning Applications,” in _AsiaCCS 2018_ .

[102] B. D. Rouhani, M. S. Riazi, and F. Koushanfar, “DeepSecure: Scalable
Provably-Secure Deep Learning,” in _DAC 2018_ .

[103] T. Ryffel, A. Trask, M. Dahl, B. Wagner, J. Mancuso, D. Rueckert,
and J. Passerat-Palmbach, “A generic framework for privacy preserving
deep learning,” _CoRR_, 2018.

[104] O. Saha, A. Kusupati, H. V. Simhadri, M. Varma, and P. Jain, “RNNPool: Efficient Non-linear Pooling for RAM Constrained Inference,”
_to appear in NeurIPS 2020_ .

[105] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L. Chen, “Mobilenetv2: Inverted residuals and linear bottlenecks,” in _CVPR 2018_ .

[106] E. Schkufza, R. Sharma, and A. Aiken, “Stochastic Optimization of
Floating-Point Programs with Tunable Precision,” in _PLDI 2014_ .

[107] A. Shamir, “How to share a secret,” _Commun. ACM_, 1979.

[108] A. Solovyev, M. S. Baranowski, I. Briggs, C. Jacobsen, Z. Rakamari´c,
and G. Gopalakrishnan, “Rigorous estimation of floating-point roundoff errors with symbolic taylor expansions,” _ACM Trans. Program._
_Lang. Syst._, 2018.

[109] E. M. Songhori, M. S. Riazi, S. U. Hussain, A.-R. Sadeghi, and
F. Koushanfar, “ARM2GC: Succinct garbled processor for secure
computation,” _arXiv 2019_ .

[110] S. Wagh, D. Gupta, and N. Chandran, “SecureNN: 3-Party Secure
Computation for Neural Network Training,” _PoPETs 2019_ .

[111] E. Wang, Q. Zhang, B. Shen, G. Zhang, X. Lu, Q. Wu, and Y. Wang,
_Intel Math Kernel Library_, 2014.

[112] P. Warden, “Speech Commands: A Dataset for Limited-Vocabulary
Speech Recognition,” _arXiv 2018_ .

[113] W.-F. Wong and E. Goto, “Fast evaluation of the elementary functions
in single precision,” _IEEE Transactions on Computers 1995_ .

[114] K. Yang, C. Weng, X. Lan, J. Zhang, and X. Wang, “Ferret: Fast
extension for correlated OT with small communication,” in _CCS_ .
ACM, 2020, pp. 1607–1626.

[115] A. C. Yao, “How to Generate and Exchange Secrets (Extended Abstract),” in _FOCS 1986_ .


A PPENDIX


_A. Optimized Protocol for F_ MUX _[ℓ]_
In this section, we present an optimized protocol for _F_ MUX _[ℓ]_
which utilizes COT and builds over the protocol used in [99].
Our optimization relies on the following observation: consider
_x ∈_ Z 2 with shares _⟨x⟩_ _[B]_ = ( _x_ 0 _, x_ 1 ) and _y ∈_ Z _L_ with shares
_⟨y⟩_ _[ℓ]_ = ( _y_ 0 _, y_ 1 ), then the following holds:


_x ∗_ _ℓ_ _y_ = ( _x_ 0 _⊕_ _x_ 1 ) _∗_ _ℓ_ ( _y_ 0 + _y_ 1 )


= ( _x_ 0 + _x_ 1 _−_ 2 _x_ 0 _∗_ _ℓ_ _x_ 1 ) _∗_ _ℓ_ ( _y_ 0 + _y_ 1 )


= _x_ 0 _∗_ _ℓ_ _y_ 0 + _x_ 1 _∗_ _ℓ_ ( _y_ 0 _−_ 2 _x_ 0 _∗_ _ℓ_ _y_ 0 )


+ _x_ 1 _∗_ _ℓ_ _y_ 1 + _x_ 0 _∗_ _ℓ_ ( _y_ 1 _−_ 2 _x_ 1 _∗_ _ℓ_ _y_ 1 )


In the above, the terms _x_ 0 _∗_ _ℓ_ _y_ 0 and _x_ 1 _∗_ _ℓ_ _y_ 1 can be locally
computed by _P_ 0 and _P_ 1, respectively, while for the other two
terms, we use � 21 �-COT _ℓ_ protocol. In particular, to calculate _̸_
shares of _x_ 1 _∗_ _ℓ_ ( _y_ 0 _−_ 2 _x_ 0 _∗_ _ℓ_ _y_ 0 ) term, _P_ 0 acts as the sender with
correlation ( _y_ 0 _−_ 2 _x_ 0 _∗_ _ℓ_ _y_ 0 ) and _P_ 1 acts as the receiver with
choice bit _x_ 1 ; similarly the term can be computed with the
sender and receiver roles reversed. Note that both the COTs

can be done in parallel giving us a 2-round solution which



communicates 2 _ℓ_ less bits than prior approach from [99] that
used 2 instances of � 21 �-OT _ℓ_ .


_B. Wrap and All Ones_

Recall that the functionality _F_ Wrap _[ℓ]_ &All1s [(] _[x, y]_ [)][ outputs]
( _⟨w⟩_ _[B]_ _||⟨e⟩_ _[B]_ ) such that _w_ = wrap( _x, y, L_ ) and _e_ = **1** _{_ ( _x_ +
_y_ mod _L_ ) = _L −_ 1 _}_ . Consider the _ℓ_ -bit functionality
_F_ _[ℓ]_
Eq [(] _[x, y]_ [)][ that returns] _[ ⟨][e][⟩]_ _[B]_ [ such that] _[ e]_ [ =] **[ 1]** _[{][x]_ [ =] _[ y][}]_ [. Then,]
_F_ Wrap _[ℓ]_ &All1s [(] _[x, y]_ [) =] _[ F]_ Mill _[ℓ]_ [(] _[L][ −]_ [1] _[ −]_ _[x, y]_ [)] _[||F]_ Eq _[ℓ]_ [(] _[L][ −]_ [1] _[ −]_ _[x, y]_ [)][,]
that is, millionaires’ and equality on the same inputs. Now,
to construct an efficient protocol for _F_ Mill _[ℓ]_ [, CrypTFlow2 [][99][]]
used the following recurrence relations: Let _x_ = ( _x_ 1 _||x_ 0 ) and
_y_ = ( _y_ 1 _||y_ 0 ) such that _x_ _i_ _, y_ _i_ _∈{_ 0 _,_ 1 _}_ _[ℓ/]_ [2] for _i ∈{_ 0 _,_ 1 _}_ . Then,


**1** _{x < y}_ = **1** _{x_ 1 _< y_ 1 _} ⊕_ ( **1** _{x_ 1 = _y_ 1 _} ∧_ **1** _{x_ 0 _< y_ 0 _}_ )


**1** _{x_ = _y}_ = **1** _{x_ 1 = _y_ 1 _} ∧_ **1** _{x_ 0 = _y_ 0 _}_


That is, they reduce the millionaires’ on _ℓ_ -bit strings to
millionaires’ and equalities on smaller strings. While they
computed millionaires’ instances on all nodes, they skipped
a small number of equality computations that were not used,
e.g. the root note. For _F_ Wrap _[ℓ]_ &All1s [, we compute millionaires’]
and equality on all notes and this marginally increases the cost
over the protocol for _F_ Mill _[ℓ]_ [. Nonetheless, the communication]
cost of _F_ Wrap _[ℓ]_ &All1s [is at most] _[ λℓ]_ [+ 14] _[ℓ]_ [.]


_C. Truncation_


_1) Proof for Lemma 1:_ For _b ∈{_ 0 _,_ 1 _}_, let _x_ _b_ = _⟨x⟩_ _[ℓ]_ _b_ [. Over]
Z, we can write _x_ _b_ = _u_ _b_ _·_ 2 _[s]_ + _v_ _b_ and have the following:


_x_ 0 + _x_ 1 = ( _v_ 0 + _v_ 1 ) + 2 _[s]_ ( _u_ 0 + _u_ 1 )

= ( _v_ 0 + _v_ 1 _−_ _c ·_ 2 _[s]_ ) + 2 _[s]_ ( _u_ 0 + _u_ 1 _−_ _d ·_ 2 _[ℓ][−][s]_ )


+ _c ·_ 2 _[s]_ + _d · L_


= _v_ _[′]_ + 2 _[s]_ ( _u_ _[′]_ + _c_ ) + _d · L_


Let _w_ _[′]_ = **1** _{u_ _[′]_ + _c >_ 2 _[ℓ][−][s]_ _−_ 1 _}_ . Then


_x_ 0 + _x_ 1 = _v_ _[′]_ + 2 _[s]_ ( _u_ _[′]_ + _c −_ _w_ _[′]_ _·_ 2 _[ℓ][−][s]_ ) + _L ·_ ( _d_ + _w_ _[′]_ ) (3)


When _d_ = 1, then _e_ = 0 and _u_ _[′]_ = _u_ 0 + _u_ 1 _−_ 2 _[ℓ][−][s]_ . Since
_u_ 0 _, u_ 1 ⩽ 2 _[ℓ][−][s]_ _−_ 1, we have that _u_ _[′]_ ⩽ 2 _[ℓ][−][s]_ _−_ 2. Therefore,
_w_ _[′]_ = 0 (because _c ∈{_ 0 _,_ 1 _}_ ). On the other hand when _d_ = 0,
_u_ _[′]_ = _u_ 0 + _u_ 1 ⩽ 2 _[ℓ][−][s]_ _−_ 1. Therefore, _w_ _[′]_ = 1 when _u_ _[′]_ =
2 _[ℓ][−][s]_ _−_ 1 (i.e., _e_ = 1) as well as _c_ = 1, and 0 otherwise.
Since at most one of _d_ and _w_ _[′]_ is 1 in any given case, we can
rewrite Equation 3 as:


_x_ 0 + _x_ 1 = _v_ _[′]_ + 2 _[s]_ ( _u_ _[′]_ + _c −_ _w_ _[′]_ _·_ 2 _[ℓ][−][s]_ ) + _L ·_ ( _d ⊕_ ( _c ∧_ _e_ ))


Since _v_ _[′]_ _<_ 2 _[s]_ and _u_ _[′]_ + _c −_ _w_ _[′]_ _·_ 2 _[ℓ][−][s]_ _<_ 2 _[ℓ][−][s]_, _w_ = _d ⊕_ ( _c ∧_ _e_ ).


_2) Division by power-of-2:_ We can write DivPow2( _x, s_ ) =
( _x≫_ _A_ _s_ ) + _m_ _x_ _∧_ _c_, where _m_ _x_ = **1** _{x_ ⩾ 2 _[ℓ][−]_ [1] _}_ is the MSB
of _x_ and _c_ = **1** _{x_ mod 2 _[s]_ = 0 _̸_ _}_ . In this equation, _m_ _x_ can
be computed with a call to _F_ Mill _[ℓ][−]_ [1] [using the integer DReLU]
protocol from [99] and _c_ can be computed with an equality
check on _s_ -bit inputs. We get _m_ _x_ _∧_ _c_ in _ℓ_ -bits with a call each
to _F_ AND and _F_ B2A _[ℓ]_ [, and then a final call to] _[ F]_ ARS _[ℓ,s]_ [gives us]
DivPow2( _x, s_ ). Since we have already computed the MSB of


**Algorithm 4** Cross Term Multiplication, Π _[m,n]_ CrossTerm [:]


**Input:** _P_ 0 holds _x ∈_ Z _M_ and _P_ 1 holds _y ∈_ Z _N_, where _m_ ⩽ _n_ .
**Output:** _P_ 0 & _P_ 1 get _⟨z⟩_ _b_ _[ℓ]_ [, where] _[ z]_ [ =] _[ x][ ∗]_ _[ℓ]_ _[y]_ [ and] _[ ℓ]_ [=] _[ m]_ [ +] _[ n]_ [.]

1: _P_ 0 parses _x_ as an _m_ -bit string _x_ = _x_ _m−_ 1 _|| · · · ||x_ 0, where
_x_ _i_ _∈{_ 0 _,_ 1 _}_ .
2: **for** _i_ = _{_ 0 _, . . ., m −_ 1 _}_ **do**
3: _P_ 0 & _P_ 1 invoke � 21 �-COT _ℓ−i_, where _P_ 0 is the sender with
input _x_ _i_ and _P_ 1 is the receiver with input _y_, and learn _⟨t_ _i_ _⟩_ _[ℓ][−][i]_ .
4: **end for**
5: For _b ∈{_ 0 _,_ 1 _}_, _P_ _b_ sets _⟨z⟩_ _b_ _[ℓ]_ [=] [�] _i_ _[m]_ =0 _[−]_ [1] [2] _[i]_ _[ · ⟨][t]_ _[i]_ _[⟩]_ _b_ _[ℓ][−][i]_ .


_x_, we employ the MSB-to-wrap optimization (Section IV-E)
here to minimize the cost of _F_ ARS _[ℓ,s]_ [. The exact cost expression]
for computing DivPow2 is given in Table V.


_D. Multiplication_


Here, we formally describe our protocols for cross term
multiplication _F_ CrossTerm _[m,n]_ [and matrix multiplication.]


_1) Cross Term Multiplication, F_ CrossTerm _[m,n]_ _[:]_ [ Our protocol for]
_F_ CrossTerm _[m,n]_ [uses COT similar to prior works [][41][], [][92][], [][99][], but]
unlike prior works, we support operands of different bitlengths.
We present our protocol in Algorithm 4 for the _m_ ⩽ _n_ case.
When _m > n_, we simply reverse the roles of the parties in
our protocol so that only _n_ COTs are performed. Correctness
of this protocol follows similarly to the prior works.



_2) Matrix Multiplication:_ Before we look at matrix multiplication, we first set some notation starting with operator
⊠ _ℓ_ : Z _[d]_ [1] _[×][d]_ [2] _×_ Z _[d]_ [2] _[×][d]_ [3] _→_ Z _[d]_ _L_ [1] _[×][d]_ [3], which does a matrix
multiplication between two input matrices _X_ and _Y_ such that
_X_ ⊠ _ℓ_ _Y_ = _X × Y_ mod _L_ . Similarly to the _∗_ _ℓ_ notation, when
one of the matrices has elements over ring Z _M_, we use the
lossless typecast operator _ζ_ _m_ to map all elements of that matrix
to Z. All the _single-input_ functionalities we consider naturally
extend to matrices, where the functionality is independently
applied to all elements of the input matrix to output a matrix of
the same dimensions. The shares of a matrix _X ∈_ Z _[d]_ _M_ [1] _[×][d]_ [2] are
denoted by _⟨X⟩_ _[m]_, where _⟨X⟩_ _[m]_ = _{⟨X_ [ _i, j_ ] _⟩_ _[m]_ _}_ _i∈_ [ _d_ 1 ] _,j∈_ [ _d_ 2 ],
and the shares of its transpose are denoted by _⟨X_ _[T]_ _⟩_ _[m]_ .
Now, consider the matrix multiplication functionality
_F_ UMatMul _[m,n,d]_ [1] _[,d]_ [2] _[,d]_ [3] that takes as input _⟨X⟩_ _[m]_ _∈_ Z _[d]_ _M_ [1] _[×][d]_ [2] and
_⟨Y ⟩_ _[n]_ _∈_ Z _[d]_ _N_ [2] _[×][d]_ [3] and outputs _⟨Z⟩_ _[ℓ]_ _∈_ Z _L_ _[d]_ [1] _[×][d]_ [3] such that
_ℓ_ = _m_ + _n_ + _⌈_ log _d_ 2 _⌉_ and _Z_ = _X_ ⊠ _ℓ_ _Y_ . As described in
Section IV-C, we need the additional _e_ = _⌈_ log _d_ 2 _⌉_ bits to
prevent integer overflow due to additions. When _m_ ⩽ _n_, we
extend the input matrix _⟨Y ⟩_ _[n]_ to get _⟨Y_ _[′]_ _⟩_ _[n]_ _[′]_ for _n_ _[′]_ = _n_ + _e_ .
Then, Equation 2 generalizes to matrices as follows:
_X_ ⊠ _ℓ_ _Y_ _[′]_ = _X_ 0 ⊠ _ℓ_ _Y_ 0 _[′]_ [+] _[ X]_ [1] [⊠] _[ℓ]_ _[Y]_ 1 _[ ′]_ [+] _[ X]_ [0] [⊠] _[ℓ]_ _[Y]_ 1 _[ ′]_ [+]
_X_ 1 ⊠ _ℓ_ _Y_ 0 _[′]_ _[−]_ [2] _[n]_ _[′]_ _[ ∗]_ _[ℓ]_ [(] _[X]_ [ ⊠] _[m]_ _[W]_ _[Y]_ _[ ′]_ [)] _[ −]_ _[M][ ∗]_ _[ℓ]_ [(] _[W]_ _[X]_ [⊠] _[n]_ _[′]_ _[ Y]_ _[ ′]_ [)][, where]
_W_ _X_ = wrap( _X_ 0 _, X_ 1 _, M_ ) and _W_ _Y_ _′_ = wrap( _Y_ 0 _[′]_ _[, Y]_ 1 _[ ′]_ _[,]_ [ 2] _[n]_ _[′]_ [)][.]
Similar to _F_ CrossTerm _[m,n]_ _[′]_ [,] we define a functionality

_F_ MatCrossTerm _[m,n]_ _[′]_ _[,d]_ [1] _[,d]_ [2] _[,d]_ [3] for matrices to compute the cross-terms
_X_ 0 ⊠ _ℓ_ _Y_ 1 _[′]_ [and] _[ X]_ [1] [⊠] _[ℓ]_ _[Y]_ 0 _[ ′]_ [. This functionality can be realized]
naively by making _d_ 1 _d_ 2 _d_ 3 independent calls to Π _[m,n]_ CrossTerm _[′]_ [.]



**Algorithm 5** Unsigned Matrix Multiplication, Π _[m,n,d]_ UMatMul [1] _[,d]_ [2] _[,d]_ [3] :


**Input:** _P_ 0 & _P_ 1 hold _⟨X⟩_ _[m]_ and _⟨Y ⟩_ _[n]_, where _X ∈_ Z _[d]_ _M_ [1] _[×][d]_ [2],
_Y ∈_ Z _[d]_ _N_ [2] _[×][d]_ [3] and _m_ ⩽ _n_ .
**Output:** _P_ 0 & _P_ 1 get _⟨Z⟩_ _[ℓ]_, where _Z_ = _X_ ⊠ _ℓ_ _Y_, _ℓ_ = _m_ + _n_ + _e_
and _e_ = _⌈_ log _d_ 2 _⌉_ .

1: _P_ 0 & _P_ 1 invoke _F_ ZExt _[n,n]_ [+] _[e]_ ( _⟨Y ⟩_ _[n]_ ) and learn _⟨Y_ _[′]_ _⟩_ _[n]_ _[′]_ .
2: For _b ∈{_ 0 _,_ 1 _}_, let _X_ _b_ = _⟨X⟩_ _b_ _[m]_ [and] _[ Y]_ _b_ _[ ′]_ [=] _[ ⟨][Y]_ _[ ′]_ _[⟩]_ _[n]_ _b_ _[′]_ [.]
3: _P_ 0 and _P_ 1 invoke the following functionalities.
4: _F_ MatCrossTerm _[m,n]_ _[′]_ _[,d]_ [1] _[,d]_ [2] _[,d]_ [3] ( _X_ 0 _, Y_ 1 _[′]_ [)][ and learn] _[ ⟨][C][⟩]_ _[ℓ]_ [.]

5: _F_ MatCrossTerm _[n]_ _[′]_ _[,m,d]_ [3] _[,d]_ [2] _[,d]_ [1] ( _Y_ _[′][T]_ 0 _[, X]_ 1 _[T]_ [)][ and learn] _[ ⟨][D][⟩]_ _[ℓ]_ [.]
6: _F_ Wrap _[m]_ [(] _[X]_ [0] _[, X]_ [1] [)][ to learn] _[ ⟨][W]_ _[X]_ _[⟩]_ _[B]_ [.]

7: _F_ Wrap _[n]_ _[′]_ [(] _[Y]_ 0 _[ ′]_ _[, Y]_ 1 _[ ′]_ [)][ to learn] _[ ⟨][W]_ _Y_ _[′]_ _[⟩]_ _[B]_ [.]

8: _F_ BitMatMul _[m,d]_ [3] _[,d]_ [2] _[,d]_ [1] ( _⟨W_ _Y_ _[T]_ _[′]_ _[⟩]_ _[B]_ _[,][ ⟨][X]_ _[T]_ _[ ⟩]_ _[m]_ [)][ to learn] _[ ⟨][G][⟩]_ _[m]_ [.]

9: _F_ BitMatMul _[n]_ _[′]_ _[,d]_ [1] _[,d]_ [2] _[,d]_ [3] ( _⟨W_ _X_ _⟩_ _[B]_ _, ⟨Y_ _[′]_ _⟩_ _[n]_ _[′]_ ) to learn _⟨H⟩_ _[n]_ _[′]_ .

10: _P_ _b_ outputs _X_ _b_ ⊠ _ℓ_ _Y_ _b_ _[′]_ [+] _[ ⟨][C][⟩]_ _[ℓ]_ _b_ [+] _[ ⟨][D]_ _[T]_ _[ ⟩]_ _[ℓ]_ _b_ _[−]_ [2] _[n]_ _[′]_ _[ ∗]_ _[ℓ]_ _[⟨][G]_ _[T]_ _[ ⟩]_ _[m]_ _b_ _[−]_
2 _[m]_ _∗_ _ℓ_ _⟨H⟩_ _b_ _[n]_ _[′]_ for _b ∈{_ 0 _,_ 1 _}_ .


Instead, we can do much better by observing that in a
matrix multiplication, each element of _X_ is multiplied with
_d_ 3 elements of _Y_ . Thus, rather than doing _d_ 3 independent
COTs on _ℓ_ _−_ _i_ bit-strings in Step 3 of Π _[m,n]_ CrossTerm _[′]_ [, we can]

perform a single COT on _d_ 3 _·_ ( _ℓ_ _−_ _i_ ) bit-strings (while
respecting the independent correlations). This method of
batching COTs was also used in prior works on secure
inference [92], [99], and it leads to an overall communication
of _d_ 1 _d_ 2 ( _mλ_ + ( _mn_ _[′]_ + _m_ [2] _/_ 2 + _m/_ 2) _d_ 3 ) bits.
Note that _⟨W_ _X_ _⟩_ _[B]_ and _⟨W_ _Y_ _′_ _⟩_ _[B]_ can be computed by making
_d_ 1 _d_ 2 calls to _F_ Wrap _[m]_ [and] _[ d]_ [2] _[d]_ [3] [ calls to] _[ F]_ Wrap _[n]_ _[′]_ [, respectively.]
Since the terms _X_ _i_ ⊠ _ℓ_ _Y_ _i_ _[′]_ [can be computed locally, the only]
terms left to compute are _X_ ⊠ _m_ _W_ _Y_ _′_ and _W_ _X_ ⊠ _n_ _′_ _Y_ _[′]_ .
They can be computed using the following functionality
_F_ BitMatMul _[ℓ,d]_ [1] _[,d]_ [2] _[,d]_ [3] [that takes a bit-matrix] _[ ⟨][W]_ _[⟩]_ _[B]_ _[ ∈{]_ [0] _[,]_ [ 1] _[}]_ _[d]_ [1] _[×][d]_ [2] [ and]
a matrix _⟨X⟩_ _[ℓ]_ _∈_ Z _[d]_ _L_ [2] _[×][d]_ [3] as inputs, and outputs a matrix
_⟨Z⟩_ _[ℓ]_ _∈_ Z _[d]_ _L_ [1] _[×][d]_ [3] such that _Z_ = _W_ ⊠ _ℓ_ _X_ . We use the OTbased MUX protocol from [99] to implement _F_ BitMatMul _[ℓ,d]_ [1] _[,d]_ [2] _[,d]_ [3] [,]
and also leverage the batching technique here to reduce the
number of OTs. The communication required by this protocol
is 2 _d_ 1 _d_ 2 ( _λ_ + 2 _ℓd_ 3 ) bits.
Our complete protocol for _F_ UMatMul _[m,n,d]_ [1] _[,d]_ [2] _[,d]_ [3] is presented in
Algorithm 5 for the _m_ ⩽ _n_ case. The total communication cost
of this protocol is _d_ 1 _d_ 2 _d_ 3 ((2 _m_ + 4)( _n_ + _e_ ) + _m_ [2] + 5 _m_ ) +
_d_ 1 _d_ 2 ( _λ_ (3 _m_ + 6) + 14 _m_ + _e −_ 6) + _d_ 2 _d_ 3 ( _λ_ ( _n_ + 2) + 14 _n_ )
bits. In the protocol, we extend _Y_ because it has elements
of larger bitwidth, and this strategy leads to better overall
communication in most cases. The other case of _m > n_ is

similar and we extend the entries of matrix _X_ by _e_ bits.


_E. Digit Decomposition_

We consider the functionality _F_ DigDec _ℓ,{d_ _i_ _}_ _i∈_ [ _c_ ] that decomposes
an _ℓ_ -bit number into _c_ sub-strings or digits of lengths _{d_ _i_ _}_
such that [�] _i∈_ [ _c_ ] _[d]_ _[i]_ [ =] _[ ℓ]_ [. More formally,] _[ F]_ DigDec _ℓ,{d_ _i_ _}_ _i∈_ [ _c_ ] takes

_⟨x⟩_ _[ℓ]_ as input and outputs _⟨z_ _c−_ 1 _⟩_ _[d]_ _[c][−]_ [1] _, . . ., ⟨z_ 0 _⟩_ _[d]_ [0] such that
_x_ = _z_ _c−_ 1 _|| . . . ||z_ 0 . We use this functionality in extracting


|Protocol|Comm. (bits)|Rounds|
|---|---|---|
|Π_m,n_<br>ZExt & Π_m,n_<br>SExt|_λ_(_m_ + 1) + 13_m_ +_ n_|log_ m_ + 2|
|_⋆_Π_m,n_<br>ZExt &_ ⋆_Π_m,n_<br>SExt|2_λ −m_ +_ n_ + 2|4|
|Π_ℓ,s_<br>LRS & Π_ℓ,s_<br>ARS|_λ_(_ℓ_+ 3) + 15_ℓ_+_ s_ + 20|log_ ℓ_+ 3|
|_⋆_Π_ℓ,s_<br>LRS &_ ⋆_Π_ℓ,s_<br>ARS|_λ_(_s_ + 3) +_ ℓ_+ 15_s_ + 2|log_ s_ + 2|
|Π_ℓ,s_<br>TR|_λ_(_s_ + 1) +_ ℓ_+ 13_s_|log_ s_ + 2|
|Π_ℓ,s_<br>DivPow2|_λ_(_ℓ_+ 7_s/_4 + 4) + 16_ℓ_+ 23_s −_5|log_ ℓ_+ 4|
|Π_m,n_<br>UMult & Π_m,n_<br>SMult|_λ_(3_µ_ +_ ν_ + 4) + 2_µν_ +_ µ_2 + 17_µ_ + 16_ν_|log_ ν_ + 2|
|_⋆_Π_m,n_<br>UMult &_ ⋆_Π_m,n_<br>SMult|_λ_(2_µ_ + 6) + 2_µν_ +_ µ_2 + 3_µ_ + 2_ν_ + 4|4|
|Π_ℓ,d_<br>DigDec|(_ℓ/d −_1)(_λ_(_d_ + 2) + 15_d_ + 20)|log_ d_ +_ ℓ/d_ + 1|
|Π_ℓ,d_<br>MSNZB|(_ℓ/d −_1)(_λ_(_d_ + 8) + 2_d_(_ι_ + 1) + 15_d_ + 2_ι_ + 60) + 6_λ_ + 2_d_(_ι_ + 1) +_ ℓ_2 + 2_ι_|log_ d_ + 2_ℓ/d_ + 7|


TABLE V: Exact communication and round expressions for our building blocks, assuming that the cost of Π _[ℓ]_ Mill [and][ Π] _[ℓ]_ Mill&Eq
is _λℓ_ + 14 _ℓ_ bits. _µ_ = min( _m, n_ ) _, ν_ = max( _m, n_ ), and _⋆_ denotes the variant of the protocol in which the MSBs of the inputs
are already known in the clear. In case the MSBs are known in the shared form, the additional cost is just _λ_ +2 bits per input.



digits to be used as input to lookup tables for approximations for exponential, initial approximation of reciprocal in
sigmoid/tanh and reciprocal square root.
For ease of exposition we first consider a simplified functionality _F_ DigDec _[ℓ,d]_ [with] _[ d][ |][ ℓ]_ [that outputs] _[ c]_ [ =] _[ ℓ/d]_ [ digits of]
equal length _d_ and present our protocol for this functionality
in Algorithm 6. Idea is as follows: To compute the shares of
_z_ _i_, it suffices to compute the carry of lower bits into this digit
when reconstructing shares of _x_ . That is, consider a parsing of
_ℓ_ -bit string _⟨x⟩_ _[ℓ]_ _b_ [as] _[ y]_ _[b,c][−]_ [1] _[||][ . . .][ ||][y]_ _[b,]_ [0] [ such that] _[ y]_ _[b,i]_ _[ ∈{]_ [0] _[,]_ [ 1] _[}]_ _[d]_

for all _i ∈_ [ _c_ ] for _b ∈{_ 0 _,_ 1 _}_ . Also, set _Y_ _b,i_ = _y_ _b,i_ _|| . . . ||y_ _b,_ 0
for all _i ∈_ [ _c_ ], _b ∈{_ 0 _,_ 1 _}_ . Now, observe that _z_ _i_ = _y_ 0 _,i_ +
_y_ 1 _,i_ + carry _i_ mod 2 _[d]_, where carry _i_ = _Y_ 0 _,i−_ 1 + _Y_ 1 _,i−_ 1 ⩾ 2 _[id]_ .
Alternatively, carry _i_ = wrap( _Y_ 0 _,i−_ 1 _, Y_ 1 _,i−_ 1 _,_ 2 _[id]_ ). In our protocol, we compute this carry _i_ using Lemma 1 iteratively (similar
to our protocol for _F_ LRS _[ℓ,s]_ [) and the variable] _[ u]_ _[i]_ [ corresponds to]
carry _i_ . The communication complexity of our protocol for the
simplified setting is ( _c −_ 1)( _λ_ ( _d_ + 2) + 15 _d_ + 20) bits.
Also, it is easy to see that the above protocol generalizes to the case of unequal size digits, by parsing
the initial shares appropriately and doing the same computation. The communication for the generalized case is
� _i∈_ [ _c−_ 1] [(] _[λ]_ [(] _[d]_ _[i]_ [ + 2) + 15] _[d]_ _[i]_ [ + 20)][ bits. In contrast, doing a]

digit-decomposition using GC would require _λ_ (6 _ℓ_ _−_ 2 _c −_ 2)
bits of communication. For example, for _ℓ_ = 32 and _d_ = 8,
our protocol has an improvement of 5 _._ 5 _×_ over GC.


_F. Most Significant Non-zero Bit (MSNZB)_


For an _ℓ_ -bit integer _x_, MSNZB( _x_ ) refers to the index of the
most significant non-zero-bit. That is, MSNZB( _x_ ) = _k ∈_ [ _ℓ_ ], if
_x_ _k_ = 1 and _x_ _j_ = 0 for all _j > k_ . Alternatively, MSNZB( _x_ ) =
_k_ if and only if 2 _[k]_ ⩽ _x <_ 2 _[k]_ [+1] . For the special case of input
being 0, MSNZB(0) = 0. Consider the functionality _F_ MSNZB _[ℓ]_
that takes as input _⟨x⟩_ _[ℓ]_ and outputs _{⟨z_ _i_ _⟩_ _[B]_ _}_ _i∈_ [ _ℓ_ ] such that
_z_ _i_ = 1 if MSNZB( _x_ ) = _i_ and 0 otherwise. Our protocol
for _F_ MSNZB _[ℓ]_ [reduces to MSNZB-like computation on integers]
on smaller bit-length as follows: For simplicity of exposition,
consider _d ∈_ N such that _d | ℓ_ . First, we invoke _F_ DigDec _[ℓ,d]_ [to]



**Algorithm 6** Digit Decomposition, Π _[ℓ,d]_ DigDec [:]


**Input:** _P_ 0 & _P_ 1 hold _⟨x⟩_ _[ℓ]_ s.t. _c_ = _ℓ/d_ .
**Output:** _P_ 0 & _P_ 1 get _{⟨z_ _i_ _⟩_ _[d]_ _}_ _i∈_ [ _c_ ] s.t. _x_ = _z_ _c−_ 1 _|| . . . ||z_ 0 .

1: For _b ∈{_ 0 _,_ 1 _}_, _P_ _b_ parses _⟨x⟩_ _b_ _[ℓ]_ [as an] _[ ℓ]_ [-bit string] _[ y]_ _[b,c][−]_ [1] _[||][ . . .][ ||][y]_ _[b,]_ [0]
s.t. _y_ _b,i_ _∈{_ 0 _,_ 1 _}_ _[d]_ for all _i ∈_ [ _c_ ].
2: For all _i ∈{_ 0 _, . . ., c−_ 2 _}_, _P_ 0 & _P_ 1 invoke _F_ Wrap _[d]_ &All1s [(] _[y]_ _[b,i]_ _[, y]_ _[b,]_ [1] [)]
and learn _⟨w_ _i_ _⟩_ _[B]_ _||⟨e_ _i_ _⟩_ _[B]_ .
3: For _b ∈{_ 0 _,_ 1 _}_, _P_ _b_ sets _⟨u_ 0 _⟩_ _b_ _[B]_ [= 0][ and] _[ ⟨][z]_ [0] _[⟩]_ _[d]_ _b_ [=] _[ y]_ _[b,]_ [0] [.]
4: **for** _i ∈{_ 1 _, . . ., c −_ 1 _}_ **do**
5: _P_ 0 & _P_ 1 invoke _F_ AND ( _⟨u_ _i−_ 1 _⟩_ _[B]_ _, ⟨e_ _i−_ 1 _⟩_ _[B]_ ) to learn _⟨v_ _i−_ 1 _⟩_ _[B]_ .
6: For _b ∈{_ 0 _,_ 1 _}_, _P_ _b_ sets _⟨u_ _i_ _⟩_ _b_ _[B]_ [=] _[ ⟨][v]_ _[i][−]_ [1] _[⟩]_ _[B]_ _b_ _[⊕⟨][w]_ _[i][−]_ [1] _[⟩]_ _[B]_ _b_ [.]
7: _P_ 0 & _P_ 1 invoke _F_ B2A _[d]_ [(] _[⟨][u]_ _[i]_ _[⟩]_ _[B]_ [)][ and learn] _[ ⟨][u]_ _[i]_ _[⟩]_ _[d]_ [.]
8: For _b ∈{_ 0 _,_ 1 _}_, _P_ _b_ sets _⟨z_ _i_ _⟩_ _b_ _[d]_ [=] _[ y]_ _[b,i]_ [+] _[ ⟨][u]_ _[i]_ _[⟩]_ _[B]_ _b_ [.]
9: **end for**


decompose _ℓ_ -bit integer _x_ into _c_ = _ℓ/d_ integers of _d_ -bits, say
_{y_ _i_ _}_ _i∈_ [ _c_ ] . Now, we compute MSNZB on each of these smaller
integers _y_ _i_ by taking into account their position _i_ in _x_ and
output an index in [ _ℓ_ ] which corresponds to MSNZB( _y_ _i_ )+ _i·d_ .
Note that MSNZB( _x_ ) = MSNZB( _y_ _i_ ) + _i · d_ if _y_ _i_ _̸_ = 0 and
_y_ _j_ = 0 for all _j > i_ . To realize this logic we also compute
whether _y_ _i_ = 0 for all _i ∈_ [ _c_ ].

More formally, let _ι_ = log _ℓ_ and consider the functionality
_F_ MSNZB _[d,ℓ,i]_ -P [for] _[ i][ ∈]_ [[] _[c]_ []][ that takes as input] _[ ⟨][y][⟩]_ _[d]_ [ and outputs]
_⟨u⟩_ _[ι]_ such that 2 _[u][−][id]_ ⩽ _y <_ 2 _[u][−][id]_ [+1] . Also, consider _F_ Zeros _[d]_
functionality that takes as input _⟨y⟩_ _[d]_ and outputs _⟨v⟩_ _[B]_ such
that _v_ = **1** _{y_ = 0 _}_ . First, our protocol invokes _F_ MSNZB _[d,ℓ,i]_ -P [on]
each of _⟨y_ _i_ _⟩_ _[d]_ (obtained from _F_ DigDec _[ℓ,d]_ [(] _[⟨][x][⟩]_ _[ℓ]_ [)][) to learn] _[ ⟨][u]_ _[i]_ _[⟩]_ _[ι]_ [.]
Next, we invoke _F_ Zeros _[d]_ [(] _[y]_ _[i]_ [)][ to learn] _[ ⟨][v]_ _[i]_ _[⟩]_ _[B]_ [. Now, for all]
_i ∈_ [ _c_ ], we compute _z_ _i_ _[′]_ [=] _[ u]_ _[i]_ _[ ·]_ [ (1] _[ ⊕]_ _[v]_ _[i]_ [)] _[ ·]_ [ �] _j>i_ _[v]_ _[j]_ [. Note]
that _z_ _i_ _[′]_ [=] _[ u]_ _[i]_ [ if] _[ y]_ _[i]_ _[ ̸]_ [= 0][ and] _[ y]_ _[j]_ [ = 0][ for all] _[ j > i]_ [ and]
0 otherwise. Moreover, at most one _z_ _i_ _[′]_ [is non-zero. Hence,]
we compute MSNZB( _x_ ) = ˜ _z_ = [�] _i_ _[z]_ _i_ _[′]_ [. Finally, to output the]

one-hot encoding described above, we invoke the functionality
_F_ One _[ℓ]_ -Hot [that takes as input] _[ ⟨][z]_ [˜] _[⟩]_ _[ι]_ [ and outputs] _[ {⟨][z]_ _[i]_ _[⟩]_ _[B]_ _[}]_ _[i][∈]_ [[] _[ℓ]_ []] [ such]
that _z_ _i_ = 1 for _i_ = ˜ _z_ and 0 otherwise. We present our protocol
for _F_ MSNZB _[ℓ]_ [in][ Algorithm 7][, for the special case of] _[ d][ |][ ℓ]_ [; it is]


**Algorithm 7** Most Significant Non-Zero Bit, Π _[ℓ,d]_ MSNZB [:]


**Input:** For _b ∈{_ 0 _,_ 1 _}_, _P_ _b_ holds _⟨x⟩_ _b_ _[ℓ]_ [,] _[ c]_ [ =] _[ ℓ/d, ι]_ [ = log] _[ ℓ]_ [.]
**Output:** For _b ∈{_ 0 _,_ 1 _}_, _P_ _b_ learns _{⟨z_ _i_ _⟩_ _b_ _[B]_ _[}]_ _i∈_ [ _ℓ_ ] [s.t.] _[ z]_ _[i]_ [= 1][ if][ 2] _[i]_ [ ⩽]
_x <_ 2 _[i]_ [+1] and 0 otherwise.

1: _P_ 0 & _P_ 1 invoke _F_ DigDec _[ℓ,d]_ [(] _[⟨][x][⟩]_ _[ℓ]_ [)][ and learn] _[ {⟨][y]_ _[i]_ _[⟩]_ _[d]_ _[}]_ _[i][∈]_ [[] _[c]_ []] [.]
2: **for** _i ∈{_ 0 _, . . ., c −_ 1 _}_ **do**
3: _P_ 0 & _P_ 1 invoke _F_ MSNZB _[d,ℓ,i]_ -P [(] _[⟨][y]_ _[i]_ _[⟩]_ _[d]_ [)][ and learn] _[ ⟨][u]_ _[i]_ _[⟩]_ _[ι]_ [.]
4: _P_ 0 & _P_ 1 invoke _F_ Zeros _[d]_ [(] _[⟨][y]_ _[i]_ _[⟩]_ _[d]_ [)][ and learn] _[ ⟨][v]_ _[i]_ _[⟩]_ _[B]_ [.]
5: For _b ∈{_ 0 _,_ 1 _}_, _P_ _b_ sets _⟨v_ _i_ _[′]_ _[⟩]_ _[B]_ _b_ [= (] _[b][ ⊕⟨][v]_ _[i]_ _[⟩]_ _[B]_ _b_ [)][.]
6: **end for**
7: _P_ 0 & _P_ 1 invoke _F_ MUX _[ι]_ [(] _[⟨][v]_ _c_ _[′]_ _−_ 1 _[⟩]_ _[B]_ _[,][ ⟨][u]_ _c−_ 1 _[⟩]_ _[ι]_ [)][ and learn] _[ ⟨][z]_ _c_ _[′]_ _−_ 1 _[⟩]_ _[ι]_ [.]
8: For _b ∈{_ 0 _,_ 1 _}_, _P_ _b_ sets _⟨w_ _c−_ 1 _⟩_ _b_ _[B]_ [=] _[ b]_ [.]
9: **for** _i ∈{c −_ 2 _, . . .,_ 0 _}_ **do**
10: _P_ 0 & _P_ 1 invoke _F_ AND ( _⟨w_ _i_ +1 _⟩_ _[B]_ _, ⟨v_ _i_ +1 _⟩_ _[B]_ ) and learn _⟨w_ _i_ _⟩_ _[B]_ .
11: _P_ 0 & _P_ 1 invoke _F_ AND ( _⟨w_ _i_ _⟩_ _[B]_ _, ⟨v_ _i_ _[′]_ _[⟩]_ _[B]_ [)][ and learn] _[ ⟨][w]_ _i_ _[′]_ _[⟩]_ _[B]_ [.]
12: _P_ 0 & _P_ 1 invoke _F_ MUX _[ι]_ [(] _[⟨][w]_ _i_ _[′]_ _[⟩]_ _[B]_ _[,][ ⟨][u]_ _[i]_ _[⟩]_ _[ι]_ [)][ and learn] _[ ⟨][z]_ _i_ _[′]_ _[⟩]_ _[ι]_ [.]
13: **end for**
14: For _b ∈{_ 0 _,_ 1 _}_, _P_ _b_ sets _⟨z_ ˜ _⟩_ _b_ _[ι]_ [=][ �] _[c]_ _i_ =0 _[−]_ [1] _[⟨][z]_ _i_ _[′]_ _[⟩]_ _[ι]_ _b_ [.]
15: _P_ 0 & _P_ 1 invoke _F_ One _[ℓ]_ -Hot [(] _[⟨][z]_ [˜] _[⟩]_ _[ι]_ [)][ and learn] _[ {⟨][z]_ _[i]_ _[⟩]_ _[B]_ _[}]_ _i∈_ [ _ℓ_ ] [.]



easy to see that the general case works in a similar manner.
Our protocol makes 1 call to _F_ DigDec _[ℓ,d]_ [,] _[ c]_ [ calls each to] _[ F]_ MSNZB _[d,ℓ,i]_ -P [,]
_F_ Zeros _[d]_ [(with] _[ i]_ [ going from 0 to] _[ c][ −]_ [1][) and] _[ F]_ MUX _[ι]_ [,][ 2] _[c][ −]_ [2][ calls]
to _F_ AND and 1 call to _F_ One _[ℓ]_ -Hot [.]


We implement both _F_ MSNZB _[d,ℓ,i]_ -P [and] _[ F]_ Zeros _[d]_ [using LUTs with]
_d_ -bit inputs. Moreover, since these are invoked on same input,
we combine them into a single LUT with entries ( _u_ _i_ _||v_ _i_ ).
Finally, we implement _F_ One _[ℓ]_ -Hot [using an LUT with] _[ ι]_ [-bit input]
and _ℓ_ -bit entries. The exact expression for communication for
_d | ℓ_ is given in Table V. The expression for the general
case can be computed similarly using expression in digit
decomposition. Based on empirical findings, we use _d_ = 8
in our implementation.


