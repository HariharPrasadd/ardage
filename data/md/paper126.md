1


# DeepJSCC-Q: Constellation Constrained Deep Joint Source-Channel Coding

### Tze-Yang Tung, David Burth Kurka, Mikolaj Jankowski, Deniz G¨und¨uz Department of Electrical and Electronics Engineering Imperial College London

**Abstract**


Recent works have shown that modern machine learning techniques can provide an alternative


approach to the long-standing joint source-channel coding (JSCC) problem. Very promising


initial results, superior to popular digital schemes that utilize separate source and channel codes,


have been demonstrated for wireless image and video transmission using deep neural networks


(DNNs). However, end-to-end training of such schemes requires a differentiable channel input


representation; hence, prior works have assumed that any complex value can be transmitted over


the channel. This can prevent the application of these codes in scenarios where the hardware


or protocol can only admit certain sets of channel inputs, prescribed by a digital constellation.


Herein, we propose _DeepJSCC-Q_, an end-to-end optimized JSCC solution for wireless image


transmission using a finite channel input alphabet. We show that _DeepJSCC-Q_ can achieve


similar performance to prior works that allow any complex valued channel input, especially


when high modulation orders are available, and that the performance asymptotically approaches


that of unconstrained channel input as the modulation order increases. Importantly, _DeepJSCC-_


_Q_ preserves the graceful degradation of image quality in unpredictable channel conditions, a


desirable property for deployment in mobile systems with rapidly changing channel conditions.


**Index Terms**


Joint source-channel coding, wireless image transmission, image compression, deep neural


networks, deep learning


This work was supported by the European Research Council (ERC) through project BEACON (No. 677854).


2



Image


Reconstruction







Fig. 1. Diagram of a typical separation-based digital image communication system.


I. Introduction


Source coding and channel coding are two essential steps in modern data transmission.


The former reduces the redundancy within the source signal, preserving essential information


needed to reconstruct the signal within a certain fidelity. For example, in image transmission,


commonly used compression schemes, such as JPEG and BPG, allow for the reduction in


communication load with minimal loss in reconstruction quality. Channel coding, on the


other hand, introduces structured redundancy to allow reliable decoding in the presence of


channel imperfections. A diagram of a typical communication system employing separate


source and channel coding is shown in Fig. 1.


It was proven by Shannon that the separation of source and channel coding is without loss


of optimality when the blocklength goes to infinity [1]. Nevertheless, in practical applications


we are limited to finite blocklengths, and it is known that combining the two coding steps,


that is, joint source-channel coding (JSCC), can achieve lower distortion for a given finite


blocklength than separate source and channel coding [2]–[4]. The most straightforward


approach to JSCC is to optimize the various parameters between source coding, channel


coding and modulation in a cross-layer framework. Although many such schemes have been


proposed over the years [5]–[7], none were able to provide sufficient gains to justify the


increased complexity.


A more fundamental approach is to redesign a JSCC scheme from scratch, directly map

ping the source signal to the modulated channel input, without conversion to bits at all.


It was shown recently in [8], that deep neural networks (DNNs) can be used to break the


complexity barrier of designing JSCC schemes for wireless image transmission. The scheme,


called _DeepJSCC_, showed appealing properties, such as lower end-to-end distortion for a


given channel blocklength compared with state-of-the-art digital compression schemes [9],


3


flexibility to adapt to different source or channel models [8], [9], ability to exploit channel


feedback [10], and capability to produce adaptive-bandwidth transmission schemes [11].


Importantly, graceful degradation of image quality with respect to decreasing channel quality


means that DeepJSCC is able to avoid the _cliff-effect_ that all separation-based schemes


suffer from; which refers to the phenomenon where the image becomes un-decodable when


the channel quality falls below a certain threshold resulting in unreliable transmission.


Another strength of DeepJSCC is that it learns a communication scheme from scratch,


optimizing all transformations in a data-driven manner using autoencoders [12] with a non

trainable differentiable channel model in the bottleneck layer. This simplifies the JSCC


design procedure, and allows adaptation to any particular source or channel domain and


quality measure. Part of that simplification stems from the fact that DeepJSCC not only


combines source and channel coding into one single mapping, but it also removes the


constellation diagrams used in digital schemes. In digital communications, channel encoded


bits are mapped to the elements of a two-dimensional finite constellation diagram, such


as quadrature amplitude modulation (QAM), phase shift keying (PSK), or amplitude shift


keying (ASK). In contrast, in DeepJSCC, the encoder can transmit arbitrary complex-valued


channel symbols, within a power constraint. However, this can hinder the adoption of Deep

JSCC in current commercial hardware and standardized protocols, which are constrained to


produce fixed sets of symbols.


In this work, we investigate the effects of constraining the transmission either to a lim

ited number of channel input symbols, or to a predefined constellation imposed externally.


This constraint can be crucial for the adoption of DeepJSCC in commercially available


hardware (e.g., radio transmitters), where modulators are hard-coded for efficiency, and


limit the output space available to the encoder. Successfully incorporating fixed channel


input constellations in DNN driven JSCC may even open the possibility of incorporating


such schemes into established standards, such as 5G telecommunications. Therefore, in this


paper we introduce a new strategy for JSCC of images, called _DeepJSCC-Q_, which allows


for the transmission of the content through fixed pre-defined constellations.


We note that the problem at hand is a JSCC problem over a discrete-input additive


white Gaussian noise (AWGN) channel. For any given finite blocklength, we have a finite


number of codewords that can be transmitted. Hence, the goal is to find the mapping from


the input images to these codewords and a matching decoder mapping that minimizes the


4


average end-to-end distortion. However, finding these mappings directly is a formidable


challenge. We formulate this problem as an end-to-end JSCC problem with a quantizer


in the middle. An encoder DNN extracts the features of the input image, which are then


quantized to the constellation points. These constellation points are transmitted over the


channel and the receiver tries to recover the input image from its noisy observations using


another DNN. Hence, the problem becomes the training of an autoencoder architecture


with a non-differentiable quantization layer followed by a non-trainable channel layer in the


middle.


The main contributions of _DeepJSCC-Q_ are:


_•_ Achieve performance close to that is achieved by the unconstrained DeepJSCC scheme


[8] even when using a highly constrained channel input representation.


_•_ Achieve superior performance compared to separate source and channel coding using


better portable graphics (BPG) codec [13] followed by low density parity check (LDPC)


codes [14].


_•_ Create a coherent mapping between the input image and constellation points, avoiding


the _cliff-effect_ present in all separation-based schemes.


_•_ Generate new constellations for a given modulation order, outperforming conventional


constellation designs.


II. Related Works


JSCC of images for wireless transmission has received numerous attention over the years.


Earliest communication systems were purely analog, where the source signal is directly


modulated unto the carrier waveform, corresponding to a most direct form of JSCC. With


the advances in digital communication and compression techniques, separation based com

munication systems became dominant. In order to overcome the limitations of the separation


approach, many works focus on optimizing the various parameters of the employed source


and channel codes with the goal of minimizing the end-to-end distortion. For example, in


[6], a general framework for matching source and channel code rates using a parametric


distortion model was proposed. Their approach is to match the source code rate to the


channel code and channel statistics in a source-rate-based optimization approach. Similarly,


in [5], a cross-layer optimization of the source code rate, channel code rate and transmitter


power for quality of service (QoS) is proposed.


5


A slightly different approach considers unequal error protection (UEP) to achieve relia

bility despite channel uncertainty. This approach typically separates the source into a base


layer and potentially multiple enhancement layers, with the base layer given the greatest


amount of error protection to ensure a baseline reconstruction quality. In [7], the source


image is split into multiple layers in the discrete wavelet transform (DWT) domain before


a quantizer and channel code is applied. The bit rate allocation between the quantizer and


channel code of each layer is optimized using an end-to-end rate distortion model. Similarly,


in [15], turbo codes [16] and Reed-Solomon codes [17] are used to achieve UEP. In [18],


instead of using different channel codes, the authors propose hierarchical modulation to


achieve UEP. However, none of these schemes were able to achieve adequate gains to justify


the increase in system complexity stemming from the joint optimization of the parameters


of multiple codes and the successive decoding needed at the receiver.


A more fundamental approach is to design the communication system from scratch,


without considering any digital interface. Different from the pure analog modulation schemes,


source signal is sampled and transmitted using modulated pulses. However, mapping the


source samples to channel inputs is in general a difficult multi-dimensional optimization


problem. Recently, it was shown that DNNs can be used to break the complexity barrier of


designing JSCC for wireless image transmission [8]. By setting up the encoder and decoder


in an autoencoder configuration with a non-trainable channel layer in between, and by


using the mean-squared error (MSE) metric as the loss function between the input and


the output, the DNN encoder was able to learn a function which maps input images to


channel inputs, and vice versa at the decoder. They demonstrated that the resultant JSCC


encoder and decoder, called _DeepJSCC_, was able to surpass the performance of JPEG2000


[19] compression followed by LDPC codes [14] for channel coding. Importantly, they also


showed that such schemes can avoid the _cliff-effect_ exhibited by all separation-based schemes,


which is when the channel quality deteriorates below the minimum channel quality to allow


successful decoding of the deployed channel code, leading to a cliff-edge drop-off in the


end-to-end performance. Since then, various works have extended this result to further


demonstrate the ability to exploit channel feedback [10] and adapt to various bandwidth


requirements without retraining [11]. In [20], the viability of DeepJSCC in an orthogonal


frequency division multiplexing (OFDM) system was shown, and in [21] it was shown that


such schemes can be extended to multi-user scenarios using the same decoder. DNN-aided


6


JSCC for wireless video transmission is studied in [22].


However, an implicit assumption in all of the works above is the ability for the commu

nication hardware to transmit arbitrary complex valued channel inputs. This may not be


true as many commercially available hardware have hard-coded standard protocols, making


these methods less viable for real world deployment. As such, in [23], an image transmission


problem is considered over a discrete input channel, where the input representation is learned


using a variational autoencoder (VAE) by assuming a Bernoulli prior. They consider the


transmission of MNIST images [24] over a binary erasure channel (BEC) and showed that


their scheme performs better than a VAE that only performs compression paired with an


LDPC code. An extension to this work [25] improved the results by using an adversarial


loss function. In contrast to these works, we investigate the transmission of natural images


over a differentiable channel model with a finite channel input alphabet.


The optimization of the input constellation of a digital communication system has also


been a long standing research challenge in information and communication theory [26]–[30].


From a channel coding perspective, the goal is to maximize the mutual information between


the channel input and output over the distribution of the channel input alphabet. This is


inline with Shannon’s theorem, which states that the capacity of the channel is defined over


the channel input distribution. Although these methods tend to use hand-crafted designs


in the past, more recently, DNNs have also been used to carry out constellation learning.


For example, in [31], the authors consider the transmission of a uniform binary source using


only a fixed set of channel input alphabet, and show that the probability distribution of


the constellation points can be learned by optimizing the bit error rate for a given channel


signal-to-noise ratio (SNR). They also showed that the constellation points themselves can


be part of the trainable parameters, resulting in even better performance. Herein, we are


concerned with optimizing the channel input constellation and distribution for natural image


sources, rather than the transmission of uniform binary data sequences.


III. Problem Statement


We consider the problem of wireless image transmission over a noisy channel, in which


communication is performed by transmitting one out of a finite set of symbols at each


channel use. An input image **x** _∈{_ 0 _, ...,_ 255 _}_ _[H][×][W]_ _[×][C]_ (where _H_, _W_ and _C_ represent the


image’s height, width and color channels, respectively) is mapped with an encoder function


7









Image


Reconstruction





Image


Reconstruction













(a) Scenario 1: both the transmitter and receiver have
full CSI knowledge.



(b) Scenario 2: the transmitter knows the noise power
while the receiver has full CSI knowledge.



Fig. 2. Overview of problem definition for both CSI scenarios.


_f_ : _{_ 0 _, ...,_ 255 _}_ _[H][×][W]_ _[×][C]_ _�→C_ _[k]_, where _C_ = _{c_ 1 _, ..., c_ _M_ _} ⊂_ C is the channel input alphabet with


_|C|_ = _M_ . We impose an average transmit power constraint _P_ [¯], such that



1

_k_



_k_
� _|z_ ¯ _i_ _|_ [2] _≤_ _P,_ [¯] (1)

_i_ =1



where ¯ _z_ _i_ is the _i_ th element of vector ¯ **z** . The channel input vector ¯ **z** is transmitted through


a noisy channel, with the transfer function **y** = Υ(¯ **z** ) = _h_ **z** ¯ + **n**, where _h ∈_ C is the channel


gain and **n** _∼_ _CN_ (0 _, σ_ [2] **I** _k×k_ ) is a complex Gaussian vector with dimensionality _k_ . Finally, a


receiver passes the channel output through a decoder function _g_ : C _[k]_ _�→{_ 0 _, ...,_ 255 _}_ _[H][×][W]_ _[×][C]_


to produce a reconstruction of the input ˆ **x** = _g_ ( **y** ).


We consider both the static and fading channel scenarios. In the former, _h_ is constant, and


it is known by both the transmitter and the receiver, corresponding to an AWGN channel.


In the case of a fading channel, we assume that _h_ takes on an independent value during


the transmission of each image, and its realization is known only by the receiver. Since the


channel state information (CSI) is known by the receiver, it can perform channel equalization


as

**y** _←_ _[h]_ _[∗]_ (2)

_|h|_ [2] **[y]** _[,]_


where _h_ _[∗]_ is the complex conjugate of _h_ . The equalized symbols are then passed to the


decoder for decoding. If the transmitter also has knowledge of _h_, as in the case of a static


channel, then the transmitter can perform precoding, such that


**z** ¯ _←_ _[h]_ _[∗]_ (3)

_|h|_ **[z]** [¯] _[.]_


8


After channel equalization at the receiver, we equivalently obtain an AWGN channel with


signal-to-noise ratio (SNR)



_dB._ (4)
�



SNR = 10 log 10



_|h|_ 2 ¯ _P_


_σ_ [2]

�



For the fading scenario, the average channel SNR is given by



_dB._ (5)
�



SNR = 10 log 10



E[ _|h|_ 2 ] ¯ _P_


_σ_ [2]

�



A diagram illustrating both scenarios is shown in Fig. 2. The _bandwidth compression ratio_


is defined as

_k_
_ρ_ = (6)
_H × W × C_ [channel symbols/pixel] _[,]_


which measures how much compression we apply to the images, with smaller number re

flecting more compression.


To measure the reconstruction quality, we use two metrics: peak signal-to-noise ratio


(PSNR) and multi-scale structural similarity index (MS-SSIM). They are defined as



dB _,_ (7)
�



PSNR( **x** _,_ ˆ **x** ) = 10 log 10



_A_ [2]
� MSE( **x** _,_ ˆ **x** )



ˆ
where MSE( **x** _,_ ˆ **x** ) = _||_ **x** _−_ **x** _||_ 2 [2] [and] _[ A]_ [ is the maximum possible value for a given pixel. For a]


24-bit RGB pixel, _A_ = 255.


The multi-scale structural similarity index (MS-SSIM) is defined as:


MS-SSIM( **x** _,_ ˆ **x** ) =


_M_

(8)

[ _l_ _M_ ( **x** _,_ ˆ **x** )] _[α]_ _[M]_ � [ _a_ _j_ ( **x** _,_ ˆ **x** )] _[β]_ _[j]_ [ _b_ _j_ ( **x** _,_ ˆ **x** )] _[γ]_ _[j]_ _,_

_j_ =1


where


_l_ _M_ ( **x** _,_ ˆ **x** ) = [2] _[µ]_ **[x]** _[µ]_ **[x]** [ˆ] [ +] _[ v]_ [1] _,_ (9)

_µ_ [2] **x** [+] _[ µ]_ [2] **x** ˆ [+] _[ v]_ [1]

_a_ _j_ ( **x** _,_ ˆ **x** ) = [2] _[σ]_ **[x]** _[σ]_ **[x]** [ˆ] [ +] _[ v]_ [2] _,_ (10)

_σ_ **x** [2] [+] _[ σ]_ **x** [2] ˆ [+] _[ v]_ [2]

_b_ _j_ ( **x** _,_ ˆ **x** ) = _σ_ _[σ]_ **x** **[x]** _σ_ **[x]** [ˆ] **x** ˆ [ +] + _[ v]_ _v_ [3] 3 _,_ (11)


_µ_ **x**, _σ_ **x** [2] [,] _[ σ]_ **x** [2] **x** ˆ [are the mean and variance of] **[ x]** [, and the covariance between] **[ x]** [ and ˆ] **[x]** [, respectively.]


_v_ 1, _v_ 2, and _v_ 3 are coefficients for numeric stability; _α_ _M_, _β_ _j_, and _γ_ _j_ are the weights for each


9









Fig. 3. Architecture of the proposed encoder and decoder models.


of the components. Each _a_ _j_ ( _·, ·_ ) and _b_ _j_ ( _·, ·_ ) is computed at a different downsampled scale


of **x** and ˆ **x** . We use the default parameter values of ( _α_ _M_, _β_ _j_, _γ_ _j_ ) provided by the original


paper [32]. MS-SSIM has been shown to perform better in approximating the human visual


perception than the more simplistic structural similarity index (SSIM) on different subjective


image and video databases.


The overall goal of our design is to characterize the encoding and decoding functions _f_


and _g_ that maximize the average reconstructed image quality, measured by either Eqn. (7)


or (8), between the input image **x** and its reconstruction at the decoder ˆ **x**, under the given


constraints on the available bandwidth ratio _ρ_, average power _P_, and constellation _C_ .


IV. Proposed Solution


Herein, we propose _DeepJSCC-Q_, a DNN-based JSCC scheme and an end-to-end training


strategy. In _DeepJSCC-Q_ we model the encoder and decoder functions as DNNs param

eterized by _**θ**_ and _**φ**_, respectively, and aim at learning the optimal parameters through


training. Rather than constraining the encoder DNN to discrete outputs, which would


require a huge output space, we will allow any output vector of dimension _k_, and employ


a “quantization” layer to map the generated latent vectors to transmitted symbols, such


that each quantization level represents a point in the constellation. We will introduce two


quantization strategies, one where the constellation _C_ is fixed, and another, where the


constellation is also part of the parameters to be trained for a given constellation order


_M_ . By making the constellation part of the trainable parameters, we can also optimize


10


the channel input geometry. As such, we separate the encoder _f_ into two stages: first a


DNN function _f_ _**θ**_ : _{_ 0 _, ...,_ 255 _}_ _[H][×][W]_ _[×][C]_ _�→_ C _[k]_ maps an input image **x** to a complex latent


representation **z** = _f_ _**θ**_ ( **x** ) before a quantizer _q_ _C_ : C _[k]_ _�→C_ _[k]_ maps the latent vector **z** to the


channel input ¯ **z** = _q_ _C_ ( **z** ).


As in previous works [8], [9], we utilize an autoencoder architecture to jointly train the


encoder and the decoder. We propose a fully convolutional encoder _f_ _**θ**_ and decoder _g_ _**φ**_


architecture as shown in Fig. 3. In the architecture, _C_ refers to the number of channels


in the output tensor of the convolution operation. _C_ out refers to the number of channels


in the final output tensor of the encoder _f_ _**θ**_, which controls the number of channel uses _k_


per image. The “Pixel shuffle” module, within the “Residual block upsample” module, is


used to increases the height and width of the input tensor by reshaping the it, such that


the channel dimensions are reduced while the height and width dimensions are increased.


This was first proposed in [33] as a less computationally expensive method for increasing


the CNN tensor dimensions without requiring large number of parameters, like tranpose


convolutional layers. The GDN layer refers to _generalised divisive normalization_, initially


proposed in [34], and has been shown to be effective in density modeling and compression


of images. The Attention layer refers to the simplified attention module proposed in [35],


which reduces the computation cost of the attention module originally proposed in [36].


The attention mechanism has been used in both [35] and [36] to improve the compression


efficiency by focusing the neural network on regions in the image that require higher bit rate.


In our model, this will allow the model to allocate channel bandwidth and power resources


optimally. _g_ _C_ refers to the two quantization strategies, which we will introduce next.


_A. Quantization_


In order to produce an encoder that outputs channel symbols from a finite constellation,


we perform quantization of the latent vector generated by the encoder, ¯ **z** = _g_ _C_ ( **z** ). We


will consider two quantization strategies: the _soft-to-hard quantizer_, first introduced by [37],


and the _learned soft-to-hard quantizer_, which is our extension of the soft-to-hard quantizer


that allows the constellation _C_ to be learned as well. We will first describe the soft-to-hard


quantizer.


_1) Soft-to-hard quantizer:_ Given the encoder output **z**, we first apply a “hard” quan

tization, which simply maps element _z_ _i_ _∈_ **z** to the nearest symbol in _C_ . This forms the


11











Fig. 4. Illustration of the soft-to-hard quantization procedure for a single value _z_ _i_ using a QPSK constellation. The
hard quatized value ¯ _z_ _i_, on the left hand side, simply maps the latent value _z_ _i_ to the nearest point in the constellation,
while the soft quantized value ˜ _z_ _i_ is the softmax weighted sum of the constellation points according to the squared _l_ 2
distance of _z_ _i_ to each constellation point.


channel input ¯ **z** . However, this operation is not differentiable. In order to obtain a differen

tiable approximation of the hard quantization operation, we will use the “soft” quantization


approach, proposed in [37]. In this approach, each quantized symbol is generated as the


softmax weighted sum of the symbols in _C_ based on their distances from _z_ _i_ ; that is,



_z_ ˜ _i_ =



_M_
�

_j_ =1



_e_ _[−][σ]_ _[q]_ _[d]_ _[ij]_
~~�~~ _Mn_ =1 _[e]_ _[−][σ]_ _[q]_ _[d]_ _[in]_ _[c]_ _[j]_ _[,]_ (12)



where _σ_ _q_ is a parameter controlling the “hardness” of the assignment, and _d_ _ij_ = _||z_ _i_ _−_ _c_ _j_ _||_ 2 [2]

is the squared _l_ 2 distance between the latent value _z_ _i_ and the constellation point _c_ _j_ . As


such, in the forward pass, the quantizer uses the hard quantization, corresponding to the


channel input ¯ **z**, and in the backward pass, the gradient from the soft quantization ˜ **z** is used


to update _**θ**_ . That is,
_∂_ **z** ¯

(13)

_∂_ **z** [=] _[ ∂]_ _∂_ **[z]** **z** [˜] _[.]_


A diagram illustrating the soft-to-hard quantizer for 4-QAM, also known as the quadrature


phase shift keying (QPSK), is shown in Fig. 4.


We consider constellation symbols that are uniformly distributed in a square lattice over


the complex plane, similar to QAM-modulation. For QAM constellation consisting of _M_


symbols, denoted as M-QAM, we define the max amplitude _A_ _max_ and inter symbol distance


12


_d_ _sym_ as:



~~�~~
~~�~~ 12 _P_ [¯]
�� ( _M_ [2] _−_ 1) _[,]_ (14)



_A_ _max_ = [(] _[M][ −]_ 2 [1][)]



_A_ _max_ = [(] _[M][ −]_ [1][)]



_d_ _sym_ =



~~�~~
~~�~~ 12 _P_ [¯]
�� ( _M_ [2] _−_ 1) _[,]_ (15)



where _P_ [¯] is the average power of the constellation under uniform distribution, i.e., E[ _C_ [2] ] =


_M_ 1 � _Mi_ =1 _[|][c]_ _[i]_ _[|]_ [2] [ = ¯] _[P]_ [.]


_2) Learned soft-to-hard quantizer:_ For the learned soft-to-hard quantization, the process


is the same except the constellation _C_ is part of the parameters to be trained. That is, given


Eqn. (12), the gradient of the soft quantized value ˜ _z_ _i_ with respect to the input is



_∂z_ ˜ _i_
= _[∂][C]_
_∂z_ _i_ _∂z_ _i_



_∂z_ ˜ _i_ (16)

_∂C_ _[.]_



The constellation points are initialized in the same way, but are allowed to be updated


during training. In order to maintain the power constraint, we normalize the symbol power


after each update



_c_ _i_ _←_



_√P_ ~~¯~~
(17)
_M_
~~��~~ _j_ =1 _[P]_ [(] _[c]_ _[j]_ [)] _[|][c]_ _[j]_ _[|]_ [2] _[c]_ _[i]_ _[.]_



Since we do not have the distribution over the constellation points _P_ ( _c_ _j_ ) _, j_ = 1 _, ..., M_,


we estimate this probability distribution with a batch of input images _{_ **x** _[v]_ _}_ _[B]_ _v_ =1 [, where] _[ B]_ [ is]


the batch size, utilizing the convex weights used in the soft assignment in Eq. (12). Since


the weights sum to 1, we can treat them as probabilities and average the probability of


each constellation point over the training batch to obtain an estimate of _P_ ( _c_ _j_ ). That is, the


probability of selecting a constellation point _c_ _j_ can be estimated as



_e_ _[−][σ]_ _[q]_ _[d]_ _ij_ _[v]_
~~�~~ _Mn_ =1 _[e]_ _[−][σ]_ _[q]_ _[d]_ _in_ _[v]_ _[,]_ (18)



_k_
�

_i_ =1



ˆ 1
_P_ ( _c_ _j_ ) =
_Bk_



_B_
�

_v_ =1



where _P_ [ˆ] ( _c_ _j_ ) is the empirical probability the constellation point _c_ _j_, estimated from a batch


of input images _{_ **x** _[v]_ _}_ _[B]_ _v_ =1 [,] _[ d]_ _[v]_ _ij_ [=] _[ ||][z]_ _i_ _[v]_ _[−]_ _[c]_ _[j]_ _[||]_ [2] 2 [is the squared] _[ l]_ [2] [distance between the] _[ i]_ [th element]


in the _v_ th latent vector in the batch and the constellation point _c_ _j_ . The constellation symbol


power is then normalized as



_c_ _i_ _←_



_√P_ ~~¯~~
(19)
_M_
~~��~~ _j_ =1 _[P]_ [ˆ][(] _[c]_ _[j]_ [)] _[|][c]_ _[j]_ _[|]_ [2] _[c]_ _[i]_ _[.]_


13


_B. Training Strategy_


In order to promote exploration of the available constellation points, we introduce a


regularization term based on the Kullback-Leibler (KL) divergence between the distribution


_P_ ( _C_ ) and a uniform distribution over the constellation set _U_ ( _C_ ). The KL divergence between


two distributions _P_ _W_ and _P_ _V_ is defined as



_P_ _W_
_D_ KL ( _P_ _W_ _|| P_ _V_ ) = E log
� � _P_ _V_



_,_ (20)
��



and it measures how different the two distributions are, with _D_ KL ( _P_ _W_ _|| P_ _V_ ) = 0 _⇐⇒_


_P_ _W_ = _P_ _V_ . By regularizing the distortion loss with the KL divergence _D_ KL ( _P_ ( _C_ ) _|| U_ ( _C_ )), we


encourage the quantizer _q_ _C_ to explore the available constellation points, which may improve


the end-to-end performance of the system. The distribution _P_ ( _C_ ) is estimated as in Eqn.


(18). Therefore, the final loss function we use for training is:


_l_ ( **x** _,_ ˆ **x** ) = _d_ ( **x** _,_ ˆ **x** ) + _λD_ KL ( _P_ [ˆ] ( _C_ ) _|| U_ ( _C_ )) _,_ (21)


where _d_ ( _·, ·_ ) is the distortion measure (MSE if evaluating on the PSNR metric or 1-MS

SSIM if evaluating on the MS-SSIM metric) and _λ_ is the weighting parameter to control the


amount of regularization.


V. Experimental Results


Herein, we perform a series of experiments to demonstrate the performance of _DeepJSCC-_


_Q_ . For the first CSI acquisition scenario, defined in Sec. III, we will consider a constant


channel gain magnitude _|h|_ = 1, which implies a static AWGN channel (referred to as


“AWGN” channel henceforth), while in the second scenario, we consider _h ∼_ _CN_ (0 _,_ 1) and


will refer to it as the “slow fading” channel. We train the model for SNR Train _∈{_ 7 _,_ 10 _,_ 16 _}_ dB


on the ImageNet dataset [38] which consists of 1.2 million RGB images of various resolution.


We split the dataset into 9 : 1 for training and validation, respectively. In order to train in


batches, we take random crops of 128 _×_ 128 from the training images. For final evaluation,


we use the Kodak dataset [1] consisting of 24 768 _×_ 512 images. We use the Pytorch [39] library


and the Adam [40] optimizer with _β_ 1 = 0 _._ 9, and _β_ 2 = 0 _._ 99 to train our encoder and decoder


networks. The learning rate was initialized at 0 _._ 0001 for the AWGN channel case, while for


1 http://r0k.us/graphics/kodak/


14


the slow fading case, we initialized at 0 _._ 00005. We use a batch size of 32 and early stopping


with a patience of 8 epochs, where the maximum number of training epochs is 1000. We


implement learning rate scheduling, where the learning rate is reduced by a factor of 0 _._ 8 if


the loss does not improve for 4 epochs consecutively. We use an average constellation power

_P_ ¯ = 1 and change the channel noise power _σ_ [2] accordingly to obtain any given SNR. The


soft-to-hard hardness parameter _σ_ _q_ is linearly annealed using the annealing function



_t_
_σ_ _q_ [(] _[t]_ [)] = min � 100 _, σ_ _q_ [(] _[t][−]_ [1)] + 5 � 10000



_,_ (22)
��



where _t_ is the parameter update step number and we initialize _σ_ _q_ [(0)] = 5. We set the weighting


for the regularizer _λ_ = 0 _._ 05 for the AWGN channel case when the size of the constellation


is relatively small, i.e., _M <_ 4096, as we empirically found it to be helpful to encourage


the channel input to be more uniformly distributed across the constellation set, while for


_M ≥_ 4096, _λ_ = 0 performed better, indicating that it is more beneficial to choose a subset


of available symbols with higher probability than using all symbols with the same frequency.


For the slow fading channel case, we found _λ_ = 0 to produce better results regardless of the


constellation order.


In order to compare the performance of our solution, we consider a baseline separation


scheme, in which images are first compressed into bits using the BPG image compression


codec [13] and then protected from channel distortion with low density parity check (LDPC)


codes. The LDPC codes we use are from the IEEE 802.11ad standard [41], with block length


672 bits for both rate 1 _/_ 2 and 3 _/_ 4 codes. We will compare the average image quality over


the evaluation dataset, with error bars showing the standard deviation of the image quality


metric across the dataset. We refer to M-ary constellations using soft-to-hard quantization as


“M-QAM”, indicating the constellation is a standard square QAM, whereas for the learned


constellations, we refer to them as “L-M”.


Fig. 5 shows the result of _DeepJSCC-Q_ trained for different SNR Train and modulation


order _M_ and tested over a range of channel SNR values for the AWGN channel case. For all


M-QAM constellations shown here, _DeepJSCC-Q_ exhibited graceful degradation of image


quality with decreasing channel quality. This is similar to the DeepJSCC results in [8], but


we are able to obtain the same behavior despite being constrained to a finite constella

tion. Moreover, when compared with the separation-based results, the _DeepJSCC-Q_ 4096

15



44 _._ 0


42 _._ 0


40 _._ 0


38 _._ 0


36 _._ 0


34 _._ 0


32 _._ 0


30 _._ 0


28 _._ 0


26 _._ 0


24 _._ 0


22 _._ 0







1 _._ 00


0 _._ 98


0 _._ 96


0 _._ 94


0 _._ 92



20 _._ 0



0 _._ 90



Fig. 5. Comparison of _DeepJSCC-Q_ trained for different SNR Train with the separation approach using BPG for


|Col1|Col2|Col3|Col4|D|DeepJSCC-<br>eepJSCC-<br>DeepJSCC-|Q 4096-Q<br>Q 64-QAM<br>Q 16-QA|AM (λ =<br>(λ = 0.<br>M (λ = 0.|0, SNRTr<br>05, SNRTr<br>05, SNRT|ain = 16d<br>ain = 10d<br>rain = 7d|B)<br>B)<br>B)|
|---|---|---|---|---|---|---|---|---|---|---|
|||||||BPG +<br>BPG +|LDPC 1<br> LDPC 1|/2 BPSK<br>/2 QPSK|||
|||||||BPG +<br>BPG +|LDPC 3<br>LDPC 1/|/4 QPSK<br>2 16-QAM|||
|0<br>2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(a) PSN~~R~~<br>Comparison of_ DeepJSCC-Q_ trained for diﬀ<br>oding and~~ L~~DPC codes for channel coding in t|0<br>2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(a) PSN~~R~~<br>Comparison of_ DeepJSCC-Q_ trained for diﬀ<br>oding and~~ L~~DPC codes for channel coding in t|0<br>2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(a) PSN~~R~~<br>Comparison of_ DeepJSCC-Q_ trained for diﬀ<br>oding and~~ L~~DPC codes for channel coding in t|0<br>2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(a) PSN~~R~~<br>Comparison of_ DeepJSCC-Q_ trained for diﬀ<br>oding and~~ L~~DPC codes for channel coding in t|0<br>2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(a) PSN~~R~~<br>Comparison of_ DeepJSCC-Q_ trained for diﬀ<br>oding and~~ L~~DPC codes for channel coding in t|0<br>2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(a) PSN~~R~~<br>Comparison of_ DeepJSCC-Q_ trained for diﬀ<br>oding and~~ L~~DPC codes for channel coding in t|0<br>2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(a) PSN~~R~~<br>Comparison of_ DeepJSCC-Q_ trained for diﬀ<br>oding and~~ L~~DPC codes for channel coding in t|0<br>2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(a) PSN~~R~~<br>Comparison of_ DeepJSCC-Q_ trained for diﬀ<br>oding and~~ L~~DPC codes for channel coding in t|0<br>2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(a) PSN~~R~~<br>Comparison of_ DeepJSCC-Q_ trained for diﬀ<br>oding and~~ L~~DPC codes for channel coding in t|0<br>2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(a) PSN~~R~~<br>Comparison of_ DeepJSCC-Q_ trained for diﬀ<br>oding and~~ L~~DPC codes for channel coding in t|0<br>2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(a) PSN~~R~~<br>Comparison of_ DeepJSCC-Q_ trained for diﬀ<br>oding and~~ L~~DPC codes for channel coding in t|



44 _._ 0


42 _._ 0


40 _._ 0


38 _._ 0


36 _._ 0


34 _._ 0


32 _._ 0


30 _._ 0


28 _._ 0


26 _._ 0


24 _._ 0


22 _._ 0







1 _._ 00


0 _._ 98


0 _._ 96


0 _._ 94


0 _._ 92



20 _._ 0



0 _._ 90

|Col1|Col2|Col3|Col4|Col5|D|Col7|DeepJSCC-<br>eepJSCC-<br>DeepJSCC-|Col9|Q 4096-Q<br>Q 64-QAM<br>Q 16-QA<br>BPG +<br>BPG +|Col11|AM (λ =<br>(λ = 0.<br>M (λ = 0.<br>LDPC 1<br>LDPC 1|Col13|0, SNRTr<br>05, SNRTr<br>05, SNRT<br>/2 BPSK<br>/2 QPSK|Col15|ain = 16d<br>ain = 10d<br>rain = 7d|Col17|B)<br>B)<br>B)|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|||||||||||||||||||
||||||||||BPG +<br>BPG +|BPG +<br>BPG +|LDPC 3<br>LDPC 1/|LDPC 3<br>LDPC 1/|/4 QPSK<br>2 16-QAM|/4 QPSK<br>2 16-QAM||||
|0<br>in <br>han|2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(b) MS-SSIM<br>with the separation approach using BPG f<br>nel case.|2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(b) MS-SSIM<br>with the separation approach using BPG f<br>nel case.|2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(b) MS-SSIM<br>with the separation approach using BPG f<br>nel case.|2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(b) MS-SSIM<br>with the separation approach using BPG f<br>nel case.|2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(b) MS-SSIM<br>with the separation approach using BPG f<br>nel case.|2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(b) MS-SSIM<br>with the separation approach using BPG f<br>nel case.|2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(b) MS-SSIM<br>with the separation approach using BPG f<br>nel case.|2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(b) MS-SSIM<br>with the separation approach using BPG f<br>nel case.|2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(b) MS-SSIM<br>with the separation approach using BPG f<br>nel case.|2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(b) MS-SSIM<br>with the separation approach using BPG f<br>nel case.|2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(b) MS-SSIM<br>with the separation approach using BPG f<br>nel case.|2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(b) MS-SSIM<br>with the separation approach using BPG f<br>nel case.|2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(b) MS-SSIM<br>with the separation approach using BPG f<br>nel case.|2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(b) MS-SSIM<br>with the separation approach using BPG f<br>nel case.|2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(b) MS-SSIM<br>with the separation approach using BPG f<br>nel case.|2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(b) MS-SSIM<br>with the separation approach using BPG f<br>nel case.|2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(b) MS-SSIM<br>with the separation approach using BPG f<br>nel case.|
|||||||||||||||||||
||||||_D_<br>_Dee_<br>_Dee_|_D_<br>_Dee_<br>_Dee_|_e_~~_e_~~_pJS_~~_C_~~_C-Q_<br>_pJSCC-Q_ <br>_pJSCC-Q_|_e_~~_e_~~_pJS_~~_C_~~_C-Q_<br>_pJSCC-Q_ <br>_pJSCC-Q_|409~~6~~-QA<br>4096-QAM<br>4096-QA<br>BPG +<br>BPG +|409~~6~~-QA<br>4096-QAM<br>4096-QA<br>BPG +<br>BPG +|~~M~~ (_λ_ = 0<br> (_λ_ = 0_._<br>M (_λ_ = 0_._<br>LDPC 1/<br>LDPC 1/|~~M~~ (_λ_ = 0<br> (_λ_ = 0_._<br>M (_λ_ = 0_._<br>LDPC 1/<br>LDPC 1/|~~,~~ SN~~R~~Trai<br>05, SNRTr<br>05, SNRT<br>2 BPSK<br>2 QPSK|~~,~~ SN~~R~~Trai<br>05, SNRTr<br>05, SNRT<br>2 BPSK<br>2 QPSK|n = 1~~6~~_dB_<br>ain = 10_d_<br>rain = 7_d_|n = 1~~6~~_dB_<br>ain = 10_d_<br>rain = 7_d_|~~)~~<br>_B_)<br>_B_)|
||||||||||BPG +<br>BPG + L|BPG +<br>BPG + L|LDPC 3/<br>DPC 1/2|LDPC 3/<br>DPC 1/2|4 QPSK<br>16-QAM|4 QPSK<br>16-QAM||||

0 2 4 6 8 10 12 14 16

SNR (dB)


(b) MS-SSIM



Fig. 6. Comparison of _DeepJSCC-Q_ using modulation order _M_ ~~=~~ ~~4~~ 096 trained for different SNR Train with the
separation approa ~~ch~~ using BPG for source coding and LDPC codes for channel coding in the AWGN channel case.

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|Col14|Col15|Col16|Col17|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|||||_D_<br>_Dee_|_D_<br>_Dee_|_e_~~_e_~~_pJS_~~_C_~~_C-Q_<br>_pJSCC-Q_|_e_~~_e_~~_pJS_~~_C_~~_C-Q_<br>_pJSCC-Q_|409~~6~~-QA<br>4096-QAM|409~~6~~-QA<br>4096-QAM|~~M~~ (_λ_ = 0<br> (_λ_ = 0_._|~~M~~ (_λ_ = 0<br> (_λ_ = 0_._|~~,~~ SN~~R~~Trai<br>05, SNRTr|~~,~~ SN~~R~~Trai<br>05, SNRTr|n = 1~~6~~_dB_<br>ain = 10_d_|n = 1~~6~~_dB_<br>ain = 10_d_|~~)~~<br>_B_)|
|||||_Dee_|_Dee_|_pJSCC-Q_|_pJSCC-Q_|4096-QA<br>BPG +<br>BPG +|4096-QA<br>BPG +<br>BPG +|M (_λ_ = 0_._<br>LDPC 1/<br>LDPC 1/|M (_λ_ = 0_._<br>LDPC 1/<br>LDPC 1/|05, SNRT<br>2 BPSK<br>2 QPSK|05, SNRT<br>2 BPSK<br>2 QPSK|rain = 7_d_|rain = 7_d_|_B_)|
|||||||||BPG +<br>BPG + L|BPG +<br>BPG + L|LDPC 3/<br>DPC 1/2|LDPC 3/<br>DPC 1/2|4 QPSK<br>16-QAM|4 QPSK<br>16-QAM||||
|0<br>2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(a) PSN~~R~~<br>Comparison of_ DeepJSCC-Q_ using modulati<br>on approa~~ch~~ using BPG for source coding and|0<br>2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(a) PSN~~R~~<br>Comparison of_ DeepJSCC-Q_ using modulati<br>on approa~~ch~~ using BPG for source coding and|0<br>2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(a) PSN~~R~~<br>Comparison of_ DeepJSCC-Q_ using modulati<br>on approa~~ch~~ using BPG for source coding and|0<br>2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(a) PSN~~R~~<br>Comparison of_ DeepJSCC-Q_ using modulati<br>on approa~~ch~~ using BPG for source coding and|0<br>2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(a) PSN~~R~~<br>Comparison of_ DeepJSCC-Q_ using modulati<br>on approa~~ch~~ using BPG for source coding and|0<br>2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(a) PSN~~R~~<br>Comparison of_ DeepJSCC-Q_ using modulati<br>on approa~~ch~~ using BPG for source coding and|0<br>2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(a) PSN~~R~~<br>Comparison of_ DeepJSCC-Q_ using modulati<br>on approa~~ch~~ using BPG for source coding and|0<br>2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(a) PSN~~R~~<br>Comparison of_ DeepJSCC-Q_ using modulati<br>on approa~~ch~~ using BPG for source coding and|0<br>2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(a) PSN~~R~~<br>Comparison of_ DeepJSCC-Q_ using modulati<br>on approa~~ch~~ using BPG for source coding and|0<br>2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(a) PSN~~R~~<br>Comparison of_ DeepJSCC-Q_ using modulati<br>on approa~~ch~~ using BPG for source coding and|0<br>2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(a) PSN~~R~~<br>Comparison of_ DeepJSCC-Q_ using modulati<br>on approa~~ch~~ using BPG for source coding and|0<br>2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(a) PSN~~R~~<br>Comparison of_ DeepJSCC-Q_ using modulati<br>on approa~~ch~~ using BPG for source coding and|0<br>2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(a) PSN~~R~~<br>Comparison of_ DeepJSCC-Q_ using modulati<br>on approa~~ch~~ using BPG for source coding and|0<br>2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(a) PSN~~R~~<br>Comparison of_ DeepJSCC-Q_ using modulati<br>on approa~~ch~~ using BPG for source coding and|0<br>2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(a) PSN~~R~~<br>Comparison of_ DeepJSCC-Q_ using modulati<br>on approa~~ch~~ using BPG for source coding and|0<br>2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(a) PSN~~R~~<br>Comparison of_ DeepJSCC-Q_ using modulati<br>on approa~~ch~~ using BPG for source coding and|0<br>2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(a) PSN~~R~~<br>Comparison of_ DeepJSCC-Q_ using modulati<br>on approa~~ch~~ using BPG for source coding and|
||||||||||||||||||



QAM model performed almost exactly on the envelope of all the separation-based schemes.


Although the _DeepJSCC-Q_ models trained with modulation orders _M_ = 16 _,_ 64 did not


perform as well as the separation-based schemes, increasing the modulation order at those


SNRs can improve the performance of _DeepJSCC-Q_, as shown in Fig. 6. We see that when


we employ a modulation order of _M_ = 4096 for SNR Train = 7 _,_ 10dB, _DeepJSCC-Q_ beats


the separation-based schemes convincingly. This shows that, the end-to-end optimization of


16



44 _._ 0


42 _._ 0


40 _._ 0


38 _._ 0


36 _._ 0


34 _._ 0


32 _._ 0


30 _._ 0


28 _._ 0


26 _._ 0


24 _._ 0


22 _._ 0







1 _._ 00


0 _._ 98


0 _._ 96


0 _._ 94


0 _._ 92


0 _._ 90


0 _._ 88



20 _._ 0

|Col1|Col2|D<br>D|DeepJSCC-<br>eepJSCC-<br>eepJSCC-|Q 4096-Q<br>Q 64-QAM<br>Q 16-QAM<br>DeepJSCC|AM (λ =<br>(λ = 0.<br>(λ = 0.<br>(¯z ∈Ck,|0, SNRTr<br>05, SNRTr<br>05, SNRTr<br>||¯z||2 2 = k|ain = 16d<br>ain = 16d<br>ain = 16d<br>)|B)<br>B)<br>B)|
|---|---|---|---|---|---|---|---|---|
|~~0~~<br>2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)|~~0~~<br>2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)|~~0~~<br>2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)|~~0~~<br>2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)|~~0~~<br>2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)|~~0~~<br>2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)|~~0~~<br>2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)|~~0~~<br>2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)|~~0~~<br>2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)|



(a) PSNR



0 _._ 86



Fig. 7. Comparison of _DeepJSCC-Q_ with constellation orders _M ∈{_ 16 _,_ 64 _,_ 4096 _}_ against non-quantized _DeepJSCC_
for the AWGN channel case.


_DeepJSCC-Q_ is fundamentally different from separation-based schemes, as the end-to-end


distortion is directly affected by the channel distortion rather than the successful decoding


of the channel code for a given source code rate.


We note that the best performance with the separation approach at each channel SNR


requires choosing the right constellation size. Increasin ~~g~~ the constellation order effectively


increases the transmission rate. However, while the increase in the transmission rate increases


the quality of the compressed image, this will also increase the error probability over the


channel. Hence, we should choose the right constellation size for each channel SNR. On the


other hand, for _DeepJSCC-Q_, given an SNR, the higher the constellation order, the better the


end-to-end performance. This can be understood from the perspective of quantization error.

|Col1|Col2|Col3|D<br>D|DeepJSCC-<br>eepJSCC-<br>eepJSCC-|Q 4096-Q<br>Q 64-QAM<br>Q 16-QAM<br>DeepJSCC|AM (λ =<br>(λ = 0.<br>(λ = 0.<br>(¯z ∈Ck,|0, SNRTr<br>05, SNRTr<br>05, SNRTr<br>||¯z||2 2 = 1|ain = 16d<br>ain = 16d<br>ain = 16d<br>)|B)<br>B)<br>B)|
|---|---|---|---|---|---|---|---|---|---|
|0<br>2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(b) MS-SSIM<br>_{_16_,_ 64_,_ 4096_}_ against non-quantized_ DeepJSC_<br>tion-based schemes, as the end-to-en<br>n rather than the successful decodin<br>ration approach at each channel SN<br>sin~~g~~ the constellation order eﬀectivel<br>rease in the transmission rate increase<br>ncrease the error probability over th<br>ion size for each channel SNR. On th<br>r the constellation order, the better th<br> the perspective of quantization erro|0<br>2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(b) MS-SSIM<br>_{_16_,_ 64_,_ 4096_}_ against non-quantized_ DeepJSC_<br>tion-based schemes, as the end-to-en<br>n rather than the successful decodin<br>ration approach at each channel SN<br>sin~~g~~ the constellation order eﬀectivel<br>rease in the transmission rate increase<br>ncrease the error probability over th<br>ion size for each channel SNR. On th<br>r the constellation order, the better th<br> the perspective of quantization erro|0<br>2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(b) MS-SSIM<br>_{_16_,_ 64_,_ 4096_}_ against non-quantized_ DeepJSC_<br>tion-based schemes, as the end-to-en<br>n rather than the successful decodin<br>ration approach at each channel SN<br>sin~~g~~ the constellation order eﬀectivel<br>rease in the transmission rate increase<br>ncrease the error probability over th<br>ion size for each channel SNR. On th<br>r the constellation order, the better th<br> the perspective of quantization erro|0<br>2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(b) MS-SSIM<br>_{_16_,_ 64_,_ 4096_}_ against non-quantized_ DeepJSC_<br>tion-based schemes, as the end-to-en<br>n rather than the successful decodin<br>ration approach at each channel SN<br>sin~~g~~ the constellation order eﬀectivel<br>rease in the transmission rate increase<br>ncrease the error probability over th<br>ion size for each channel SNR. On th<br>r the constellation order, the better th<br> the perspective of quantization erro|0<br>2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(b) MS-SSIM<br>_{_16_,_ 64_,_ 4096_}_ against non-quantized_ DeepJSC_<br>tion-based schemes, as the end-to-en<br>n rather than the successful decodin<br>ration approach at each channel SN<br>sin~~g~~ the constellation order eﬀectivel<br>rease in the transmission rate increase<br>ncrease the error probability over th<br>ion size for each channel SNR. On th<br>r the constellation order, the better th<br> the perspective of quantization erro|0<br>2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(b) MS-SSIM<br>_{_16_,_ 64_,_ 4096_}_ against non-quantized_ DeepJSC_<br>tion-based schemes, as the end-to-en<br>n rather than the successful decodin<br>ration approach at each channel SN<br>sin~~g~~ the constellation order eﬀectivel<br>rease in the transmission rate increase<br>ncrease the error probability over th<br>ion size for each channel SNR. On th<br>r the constellation order, the better th<br> the perspective of quantization erro|0<br>2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(b) MS-SSIM<br>_{_16_,_ 64_,_ 4096_}_ against non-quantized_ DeepJSC_<br>tion-based schemes, as the end-to-en<br>n rather than the successful decodin<br>ration approach at each channel SN<br>sin~~g~~ the constellation order eﬀectivel<br>rease in the transmission rate increase<br>ncrease the error probability over th<br>ion size for each channel SNR. On th<br>r the constellation order, the better th<br> the perspective of quantization erro|0<br>2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(b) MS-SSIM<br>_{_16_,_ 64_,_ 4096_}_ against non-quantized_ DeepJSC_<br>tion-based schemes, as the end-to-en<br>n rather than the successful decodin<br>ration approach at each channel SN<br>sin~~g~~ the constellation order eﬀectivel<br>rease in the transmission rate increase<br>ncrease the error probability over th<br>ion size for each channel SNR. On th<br>r the constellation order, the better th<br> the perspective of quantization erro|0<br>2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(b) MS-SSIM<br>_{_16_,_ 64_,_ 4096_}_ against non-quantized_ DeepJSC_<br>tion-based schemes, as the end-to-en<br>n rather than the successful decodin<br>ration approach at each channel SN<br>sin~~g~~ the constellation order eﬀectivel<br>rease in the transmission rate increase<br>ncrease the error probability over th<br>ion size for each channel SNR. On th<br>r the constellation order, the better th<br> the perspective of quantization erro|0<br>2<br>4<br>6<br>8<br>10<br>12<br>14<br>16<br>SNR (dB)<br>(b) MS-SSIM<br>_{_16_,_ 64_,_ 4096_}_ against non-quantized_ DeepJSC_<br>tion-based schemes, as the end-to-en<br>n rather than the successful decodin<br>ration approach at each channel SN<br>sin~~g~~ the constellation order eﬀectivel<br>rease in the transmission rate increase<br>ncrease the error probability over th<br>ion size for each channel SNR. On th<br>r the constellation order, the better th<br> the perspective of quantization erro|



A higher order modulation essentially corresponds to greater number of quantization levels


of the encoder output **z** = _f_ _**θ**_ ( **x** ), and thus a greater accuracy of ¯ **z** in representing **z** . Since


DeepJSCC has already been shown to surpass the performance of BPG and LDPC codes by


[9], naturally, as we increase the constellation order, the performance of _DeepJSCC-Q_ will


approach that of DeepJSCC. This is also why _DeepJSCC-Q_ with 4096-QAM constellation


performs better than 64-QAM in Fig. 5 for SNR _>_ 7, even when the 4096-QAM model was


trained for a higher SNR than the 64-QAM model.


This observation is further supported by Fig. 7, where we can see that increasing the


17








































18

































the large size of the constellation ma ~~k~~ es it challenging to further improve its performance.


This is supported by Fig. 7, where 4096-QAM performs nearly as well as the non-quantized


result. Although the L-4096 result should perform at least as good as the 4096-QAM result,


it is possible that the objective function leads to large changes to the constellation that


produces suboptimal results. We believe that better tuning of the hyperparameters, such


as the quantization hardness parameter _σ_ _q_, may alleviate this issue. A similar pattern can


be seen for the slow fading channel case shown in Fig. 9, where the margins between the


learned and QAM constellations are much bigger, with the learned constellations producing


substantially better results than their QAM counterparts. In fact, the L-64 results even


outperformed the 4096-QAM results in the PSNR metric. This shows the importance of


optimizing channel input geometry and distribution for non-Gaussian channels.


To understand the type of constellations learned by _DeepJSCC-Q_ using learned soft-to

hard quantization, we refer to Fig. 10, where we show the visualization of the constellation


points and the probability of selecting each of the points for the AWGN channel case. We


can see that for a low modulation order, such as _M_ = 4 shown in Fig. 10a and 10b, the


constellation points learned by _DeepJSCC-Q_ are not very different from 4-QAM, and the


distribution across the points is close to uniform. However, when we increase the modulation


order to _M_ = 16, we begin to see significant differences between L-16 and 16-QAM, as


shown by Fig. 10c and 10d. The L-16 constellation is clearly non-square and not centered


19



0 _._ 23 0 _._ 25 0 _._ 28 0 _._ 3

ˆ
_P_ ( _C_ )


0 _._ 5


0


_−_ 0 _._ 5


_−_ 0 _._ 5 0 0 _._ 5


In-phase


(a) 4-QAM


5 _·_ 10 _[−]_ [2] 0 _._ 1

ˆ
_P_ ( _C_ )


1


0 _._ 5


0


_−_ 0 _._ 5


_−_ 1
_−_ 1 _−_ 0 _._ 5 0 0 _._ 5 1


In-phase


(c) 16-QAM



0 _._ 23 0 _._ 25 0 _._ 28

ˆ
_P_ ( _C_ )


0 _._ 5


0


_−_ 0 _._ 5


_−_ 0 _._ 5 0 0 _._ 5


In-phase


(b) L-4


5 _·_ 10 _[−]_ [2] [ 0] _[.]_ [1] 0 _._ 15

ˆ
_P_ ( _C_ )


1 _._ 5

1


0 _._ 5

0


_−_ 0 _._ 5

_−_ 1
_−_ 1 _−_ 0 _._ 5 0 0 _._ 5 1 1 _._ 5


In-phase


(d) L-16



Fig. 10. Probability distribution of constellation points for fixed and learned constellations in the AWGN channel

case.


about the origin (0 _,_ 0), with constellation points in the first quadrant having much higher


power than those in the third quadrant. This asymmetry may be part of the reason for the


performance gain as the channel noise is zero mean and symmetric, meaning using the high


power constellation points in the first quadrant can increase the instantaneous SNR of the


received signal. The average power is then maintained by choosing the constellations in the


third quadrant much more frequently than the first quadrant. Note that this observation can


be seen as a form of unequal error protection (UEP) as well, with few important features


using high power constellation points (first quadrant) and less important features using lower


power constellation points (third quadrant).


For the slow fading case, we also see substantial differences between L-16 and 16-QAM,


as shown in Fig. 11. While both the L-16 and the 16-QAM constellations select the central


20



0 _._ 1 0 _._ 2

ˆ
_P_ ( _C_ )


1


0 _._ 5


0


_−_ 0 _._ 5


_−_ 1
_−_ 1 _−_ 0 _._ 5 0 0 _._ 5 1


In-phase


(a) 16-QAM



5 _·_ 10 _[−]_ [2] [ 0] _[.]_ [15] 0 _._ 25

ˆ
_P_ ( _C_ )


1 _._ 5

1


0 _._ 5

0


_−_ 0 _._ 5

_−_ 1
_−_ 1 _−_ 0 _._ 5 0 0 _._ 5 1 1 _._ 5


In-phase


(b) L-16



Fig. 11. Probability distribution of constellation points for fixed and learned constellations using modulation order
_M_ = 16 in the slow fading channel case.


constellation points more frequently in the slow fading channel, the L-16 constellation is


much more circular. The L-16 constellation exhibits two circles around a center point. The


outer constellation points also have more power than even the highest power constellation


points in 16-QAM, with the average power maintained by radially decreasing the probability


distribution. Moreover, when compared with the L-16 constellation learned under the AWGN


channel, Fig. 11b shows a much more centered constellation, when compared to Fig. 10d.


This is similar to the results from [31], but here we are showing these properties in a JSCC


context.


Lastly, we investigate the relationship between the _bandwidth compression ratio ρ_ and


the reconstruction distortion. Fig. 12 compares the performance of _DeepJSCC-Q_ using


_M_ = 64 with separate source and channel coding for SNR = 10dB for the AWGN channel


case. This is akin to plotting the end-to-end rate distortion performance for each of the


schemes considered herein. We see that in all but one instance, _DeepJSCC-Q_ performs


better than separation using BPG for source coding and an LDPC code for channel coding,


demonstrating the superiority of JSCC in the finite block length regime. Visual examples of


the schemes considered herein given different bandwidth compression ratios _ρ_ are presented


in Fig. 13, where the last column of images clearly show that BPG produces more blurry


images when compared to _DeepJSCC-Q_ using both L-64 and 64QAM constellations, under


_ρ_ = 1 _/_ 48. The only instance where _DeepJSCC-Q_ performed worse than separation is when


the constellation is restricted to 64-QAM and the bandwidth compression ratio is _ρ_ = 1 _/_ 6.


21



42 _._ 0


40 _._ 0


38 _._ 0


36 _._ 0


34 _._ 0


32 _._ 0


30 _._ 0


28 _._ 0


26 _._ 0





1 _._ 00


0 _._ 98


0 _._ 96


0 _._ 94


0 _._ 92


0 _._ 90


0 _._ 88



0 _._ 86

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|
|---|---|---|---|---|---|---|---|
|||||||||
|||||||||
|||||||||
|||||||||
|||||||||
|||||||||
|_D_|_D_|||||||
|_D_|_D_|_D_|_epJSCC-Q_ 64-QAM (_λ_|_epJSCC-Q_ 64-QAM (_λ_|= 0_._05, SNRTrain =|10_dB_)|10_dB_)|
||||_DeepJSCC-Q_ L-64 (_λ_ =<br>BPG LDP|_DeepJSCC-Q_ L-64 (_λ_ =<br>BPG LDP|0_._05, SNRTrain = 1<br>C 1/2 + QPSK|0_dB_)|0_dB_)|
|0_._05<br>0_._10<br>0_._15<br>_ρ_|0_._05<br>0_._10<br>0_._15<br>_ρ_|0_._05<br>0_._10<br>0_._15<br>_ρ_|0_._05<br>0_._10<br>0_._15<br>_ρ_|0_._05<br>0_._10<br>0_._15<br>_ρ_|0_._05<br>0_._10<br>0_._15<br>_ρ_|0_._05<br>0_._10<br>0_._15<br>_ρ_|0_._05<br>0_._10<br>0_._15<br>_ρ_|



(b) MS-SSIM





24 _._ 0

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|
|---|---|---|---|---|---|---|---|
|||||||||
|||||||||
|||||||||
|||||||||
|||||||||
|||||||||
|||||||||
|||||||||
||_Dee_<br>|_Dee_<br>|_pJSCC-Q_ 64-QAM (_λ_<br>_eepJSCC-Q_ L-64 (_λ_|_pJSCC-Q_ 64-QAM (_λ_<br>_eepJSCC-Q_ L-64 (_λ_|= 0_._05, SNRTrain =<br> 0_._05, SNR= 1|10_dB_)<br>_dB_)|10_dB_)<br>_dB_)|
||||BPG LDP|BPG LDP|Train<br>C 1/2 + QPSK|||

0 _._ 05 0 _._ 10 0 _._ 15


_ρ_


(a) PSNR



Fig. 12. Comparison of _DeepJSCC-Q_ for different bandwidth compression ratio _ρ_ for the AWGN channel case.


BPG


L-64


64QAM


Fig. 13. Visualization of image samples for different bandwidth compression ratios transmitted over the AWGN
channel.


This is likely because, as the bandwidth utilization increases, the encoder has greater degrees


of freedom, allowing for more expressive channel input features that benefit from non-uniform


quantization.


22


VI. Conclusions


In this paper, we have proposed _DeepJSCC-Q_, an end-to-end optimized joint source

channel coding scheme for wireless image transmission that is able to utilize a fixed channel


input constellation and achieve similar performance to unquantized DeepJSCC, as previously


proposed in [8]. Even with a constrained constellation, we are able to achieve superior per

formance to separation-based schemes using BPG for source coding and LDPC for channel


coding, all the while avoiding the _cliff-effect_ that plagues the separation-based schemes.


This makes the viability of _DeepJSCC-Q_ in existing commercial hardware with standardized


protocols much more attractive. We also show that with sufficiently high modulation order,


_DeepJSCC-Q_ can approach the performance of DeepJSCC, which does not have a fixed


channel input constellation. As such, if such constellations are available on the hardware,


_DeepJSCC-Q_ can perform nearly as well as DeepJSCC. Finally, we demonstrate that it


is possible to learn a finite channel input alphabet that further improves the performance


of _DeepJSCC-Q_, with the resultant constellation showing highly non-trivial geometry and


distribution. These promising results can bring DNN-based JSCC schemes closer to real


world deployment.


References


[1] C. E. Shannon, “A mathematical theory of communication,” _Bell Syst. Tech. J._, vol. 27, pp. 379–423 and 623–656,


July and October 1948.


[2] R. G. Gallager, _Information Theory and Reliable Communication_ . USA: John Wiley & Sons, Inc., 1968.


[3] V. Kostina and S. Verd´u, “Lossy joint source-channel coding in the finite blocklength regime,” _IEEE Transactions_


_on Information Theory_, vol. 59, pp. 2545–2575, May 2013.


[4] M. Gastpar, B. Rimoldi, and M. Vetterli, “To code, or not to code: Lossy source-channel communication


revisited,” _IEEE Transactions on Information Theory_, vol. 49, pp. 1147–1158, May 2003.


[5] W. Yu, Z. Sahinoglu, and A. Vetro, “Energy efficient JPEG 2000 image transmission over wireless sensor


networks,” in _IEEE Global Telecommunications Conference, 2004. GLOBECOM ’04._, vol. 5, pp. 2738–2743


Vol.5, Nov. 2004.


[6] S. Appadwedula, D. Jones, K. Ramchandran, and L. Qian, “Joint source channel matching for a wireless image


transmission,” in _Proceedings 1998 International Conference on Image Processing. ICIP98 (Cat. No.98CB36269)_,


vol. 2, pp. 137–141 vol.2, Oct. 1998.


[7] J. Cai and C. W. Chen, “Robust joint source-channel coding for image transmission over wireless channels,”


_IEEE Transactions on Circuits and Systems for Video Technology_, vol. 10, pp. 962–966, Sept. 2000.


[8] E. Bourtsoulatze, D. Burth Kurka, and D. G¨und¨uz, “Deep joint source-channel coding for wireless image


transmission,” _IEEE Transactions on Cognitive Communications and Networking_, vol. 5, pp. 567–579, Sep. 2019.


23


[9] D. Burth Kurka and D. G¨und¨uz, “Joint source-channel coding of images with (not very) deep learning,” in


_International Zurich Seminar on Information and Communication (IZS 2020). Proceedings_, pp. 90–94, ETH


Zurich, 2020.


[10] D. B. Kurka and D. G¨und¨uz, “Deepjscc-f: Deep joint source-channel coding of images with feedback,” _IEEE_


_Journal on Selected Areas in Information Theory_, vol. 1, no. 1, pp. 178–193, 2020.


[11] D. B. Kurka and D. G¨und¨uz, “Bandwidth-agile image transmission with deep joint source-channel coding,”


_IEEE Transactions on Wireless Communications_, pp. 1–1, 2021.


[12] D. H. Ballard, “Modular learning in neural networks,” in _Proceedings of the Sixth National Conference on Artificial_


_Intelligence - Volume 1_, AAAI’87, p. 279–284, AAAI Press, 1987.


[13] F. Bellard, _Better Portable Graphics_ [, 2014 (accessed March 13, 2020). https://bellard.org/bpg/.](https://bellard.org/bpg/)


[14] R. Gallager, “Low-density parity-check codes,” _IRE Transactions on Information Theory_, vol. 8, pp. 21–28, Jan.


1962. Conference Name: IRE Transactions on Information Theory.


[15] N. Thomos, N. Boulgouris, and M. Strintzis, “Wireless image transmission using turbo codes and optimal unequal


error protection,” _IEEE Transactions on Image Processing_, vol. 14, pp. 1890–1901, Nov. 2005.


[16] C. Berrou and A. Glavieux, “Near optimum error correcting coding and decoding: Turbo-codes,” _IEEE_


_Transactions on Communications_, vol. 44, pp. 1261–1271, Oct. 1996.


[17] S. Lin and D. J. Costello, _Error control coding: fundamentals and applications_ . Upper Saddle River, NJ:


Pearson/Prentice Hall, 2004.


[18] S. S. Arslan, P. C. Cosman, and L. B. Milstein, “Coded hierarchical modulation for wireless progressive image


transmission,” _IEEE Transactions on Vehicular Technology_, vol. 60, pp. 4299–4313, Nov. 2011.


[19] C. Christopoulos, A. Skodras, and T. Ebrahimi, “The JPEG2000 still image coding system: an overview,” _IEEE_


_Transactions on Consumer Electronics_, vol. 46, pp. 1103–1127, Nov. 2000.


[20] M. Yang, C. Bian, and H.-S. Kim, “Deep joint source channel coding for wireless image transmission with


OFDM,” _arXiv:2101.03909 [cs, eess, math]_, May 2021.


[21] M. Ding, J. Li, M. Ma, and X. Fan, “SNR-adaptive deep joint source-channel coding for wireless image


transmission,” in _ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing_


_(ICASSP)_, pp. 1555–1559, June 2021. ISSN: 2379-190X.


[22] T.-Y. Tung and D. G¨und¨uz, “DeepWiVe: Deep-learning-aided wireless video transmission,” _arXiv:2111.13034_


_[cs, eess]_, Nov. 2021.


[23] K. Choi, K. Tatwawadi, T. Weissman, and S. Ermon, “NECST: Neural joint source-channel coding,” Sept. 2018.


[24] L. Deng, “The MNIST database of handwritten digit images for machine learning research [Best of the Web],”


_IEEE Signal Processing Magazine_, vol. 29, pp. 141–142, Nov. 2012.


[25] Y. Song, M. Xu, L. Yu, H. Zhou, S. Shao, and Y. Yu, “Infomax neural joint source-channel coding via adversarial


bit flip,” _Proceedings of the AAAI Conference on Artificial Intelligence_, vol. 34, pp. 5834–5841, Apr. 2020.


[26] F. Kayhan and G. Montorsi, “Constellation design for transmission over nonlinear satellite channels,” in _2012_


_IEEE Global Communications Conference (GLOBECOM)_, pp. 3401–3406, Dec. 2012.


[27] F. Kayhan and G. Montorsi, “Constellation design for channels affected by phase noise,” in _2013 IEEE_


_International Conference on Communications (ICC)_, pp. 3154–3158, June 2013.


[28] G. Foschini, R. Gitlin, and S. Weinstein, “Optimization of two-dimensional signal constellations in the presence


of Gaussian noise,” _IEEE Transactions on Communications_, vol. 22, pp. 28–38, Jan. 1974.


[29] G. J. Foschini, R. D. Gitlin, and S. B. Weinstein, “On the selection of a two-dimensional signal constellation


24


in the presence of phase jitter and Gaussian noise,” _Bell System Technical Journal_, vol. 52, no. 6, pp. 927–965,


1973.


[30] A. J. Kearsley, “Global and local optimization algorithms for optimal signal set design,” _Journal of Research of_


_the National Institute of Standards and Technology_, vol. 106, no. 2, pp. 441–454, 2001.


[31] F. A. Aoudia and J. Hoydis, “Joint learning of probabilistic and geometric shaping for coded modulation


systems,” in _2020 IEEE Global Communications Conference (GLOBECOM)_, pp. 1–6, Dec. 2020.


[32] Z. Wang, E. P. Simoncelli, and A. C. Bovik, “Multiscale structural similarity for image quality assessment,”


in _The Thrity-Seventh Asilomar Conference on Signals, Systems Computers, 2003_, vol. 2, pp. 1398–1402 Vol.2,


Nov. 2003.


[33] W. Shi, J. Caballero, F. Husz´ar, J. Totz, A. P. Aitken, R. Bishop, D. Rueckert, and Z. Wang, “Real-time


single image and video super-resolution using an efficient sub-pixel convolutional neural network,” in _2016 IEEE_


_Conference on Computer Vision and Pattern Recognition (CVPR)_, pp. 1874–1883, June 2016.


[34] J. Ball´e, V. Laparra, and E. P. Simoncelli, “Density modeling of images using a generalized normalization


transformation,” _arXiv preprint arXiv:1511.06281_, 2015.


[35] Z. Cheng, H. Sun, M. Takeuchi, and J. Katto, “Learned image compression with discretized gaussian mixture


likelihoods and attention modules,” in _2020 IEEE Conference on Computer Vision and Pattern Recognition_


_(CVPR)_, pp. 7936–7945, June 2020.


[36] X. Wang, R. Girshick, A. Gupta, and K. He, “Non-Local Neural Networks,” in _2018 IEEE Conference on_


_Computer Vision and Pattern Recognition (CVPR)_, pp. 7794–7803, June 2018.


[37] E. Agustsson, F. Mentzer, M. Tschannen, L. Cavigelli, R. Timofte, L. Benini, and L. V. Gool, “Soft-to-hard vector


quantization for end-to-end learning compressible representations,” in _2017 Advances in Neural Information_


_Processing Systems (NIPS)_, pp. 1141–1151, Dec. 2017.


[38] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “Imagenet: A large-scale hierarchical image


database,” in _2009 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_, pp. 248–255, June


2009.


[39] A. Paszke, S. Gross, S. Chintala, G. Chanan, E. Yang, Z. DeVito, Z. Lin, A. Desmaison, L. Antiga, and A. Lerer,


“Automatic differentiation in PyTorch,” Oct. 2017.


[40] D. P. Kingma and J. Ba, “Adam: A Method for Stochastic Optimization,” _arXiv:1412.6980 [cs]_, Jan. 2017.


arXiv: 1412.6980.


[41] “IEEE standard for information technology–Telecommunications and information exchange between sys

tems–Local and metropolitan area networks–Specific requirements-Part 11: Wireless LAN medium access control


(MAC) and physical layer (PHY) specifications amendment 3: Enhancements for very high throughput in the 60


GHz band,” _IEEE Std 802.11ad-2012 (Amendment to IEEE Std 802.11-2012, as amended by IEEE Std 802.11ae-_


_2012 and IEEE Std 802.11aa-2012)_, pp. 1–628, Dec. 2012.


