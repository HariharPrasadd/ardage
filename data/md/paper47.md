## D EEP L EARNING FOR T IME S ERIES F ORECASTING : T HE E LECTRIC L OAD C ASE



**Alberto Gasparin** **Slobodan Lukovic**
Faculty of Informatics Faculty of Informatics
Università della Svizzera Italiana Università della Svizzera Italiana
6900 Lugano, Switzerland 6900 Lugano, Switzerland
```
 alberto.gasparin@usi.ch slobodan.lukovic@usi.ch
```

**Cesare Alippi**
Dept. of Electronics, Information, and Bioengineering
Politecnico di Milano
20133 Milan, Italy,
Faculty of Informatics
Università della Svizzera Italiana
6900 Lugano, Switzerland
```
           cesare.alippi@polimi.it

```


**A** **BSTRACT**


Management and efficient operations in critical infrastructure such as Smart Grids take huge advantage
of accurate power load forecasting which, due to its nonlinear nature, remains a challenging task.
Recently, deep learning has emerged in the machine learning field achieving impressive performance
in a vast range of tasks, from image classification to machine translation. Applications of deep learning
models to the electric load forecasting problem are gaining interest among researchers as well as the
industry, but a comprehensive and sound comparison among different architectures is not yet available
in the literature. This work aims at filling the gap by reviewing and experimentally evaluating on two
real-world datasets the most recent trends in electric load forecasting, by contrasting deep learning
architectures on short term forecast (one day ahead prediction). Specifically, we focus on feedforward
and recurrent neural networks, sequence to sequence models and temporal convolutional neural
networks along with architectural variants, which are known in the signal processing community but
are novel to the load forecasting one.


_**K**_ **eywords** smart grid, electric load forecasting, time-series prediction, deep learning, recurrent neural network, lstm,
gru, temporal convolutional neural network, sequence to sequence models


**1** **Introduction**


Smart grids aim at creating automated and efficient energy delivery networks which improve power delivery reliability
and quality, along with network security, energy efficiency, and demand-side management aspects [1]. Modern power
distribution systems are supported by advanced monitoring infrastructures that produce immense amount of data, thus
enabling fine grained analytics and improved forecasting performance. In particular, electric load forecasting emerges as
a critical task in the energy field, as it enables useful support for decision making, supporting optimal pricing strategies,
seamless integration of renewables and maintenance cost reductions. Load forecasting is carried out at different time
horizons, ranging from milliseconds to years, depending on the specific problem at hand.


In this work we focus on the day-ahead prediction problem also referred in the literature as _short term load forecasting_
(STLF) [2]. Since deregulation of electric energy distribution and wide adoption of renewables strongly affects daily
market prices, STLF emerges to be of fundamental importance for efficient power supply [3]. Furthermore, we
differentiate forecasting on the granularity level at which it is applied. For instance, in individual household scenario,
load prediction is rather difficult as power consumption patterns are highly volatile. On the contrary, aggregated load
consumption i.e., that associated with a neighborhood, a region, or even an entire state, is normally easier to predict as
the resulting signal exhibit slower dynamics.


A PREPRINT


**Reference** **Predictive Family of Models** **Time Horizon** **Exogenous Variables** **Dataset (Location)**

[18] LSTM, GRU, ERNN, NARX, ESN D   - Rome, Italy

[18] LSTM, GRU, ERNN, NARX, ESN D T New England [27]

[28] ERNN H T, H, P, other _[∗]_ Palermo, Italy

[17] ERNN H T, W, H Hubli, India

[20] ESN 15min to 1Y   - Sceaux, France [29]

[21] LSTM, NARX D   - Unknown

[22] LSTM D(?) C, TI Australia [30]

[23] LSTM 2W to 4M T, W, H, C, TI France

[31] LSTM 2D T, P, H, C, TI Unknown [32]


[33] LSTM, seq2seq-LSTM 60 H C, TI Sceaux, France [29]

[34] LSTM, seq2seq-LSTM 12 H T, C, TI New England [35]

[24] GRU D T, C, other _[∗∗]_ Dongguan, China


[15] CNN D C, TI USA

[16] CNN D C, TI Sceaux, France

[25] CNN + LSTM D T, C, TI North-China

[26] CNN + LSTM D   - North-Italy


Table 1: _Time Horizon:_ H(hour), D (day), W(week), M(month), Y(year), ?(Not explicitly stated, thus, inferred from
text) _Exogenous variables:_ T (temperature), W (wind speed), H (humidity), P (pressure), C (calendar including date and
holidays information), TI (time), _[∗]_ other input features were created for this dataset, _[∗∗]_ categorical weather information
is used (e.g., sunny, cloudy), _Dataset_ : the data source, a link is provided whenever available.


Historical power loads are time-series affected by several external time-variant factors, such as weather conditions,
human activities, temporal and seasonal characteristics that make their predictions a challenging problem. A large
variety of prediction methods has been proposed for the electric load forecasting over the years and, only the most
relevant ones are reviewed in this section. Autoregressive moving average models (ARMA) were among the first model
families used in short-term load forecasting [4,5]. Soon they were replaced by ARIMA and seasonal ARIMA models [6]
to cope with time variance often exhibited by load profiles. In order to include exogenous variables like temperature into
the forecasting method, model families were extended to ARMAX [7,8] and ARIMAX [9]. The main shortcoming of
these system identification families is the linearity assumption for the system being observed, hypothesis that does not
generally hold. In order to solve this limitation, nonlinear models like Feed Forward Neural Networks were proposed

–
and became attractive for those scenarios exhibiting significant nonlinearity, as in load forecasting tasks [3,10 13].
The intrinsic sequential nature of time series data was then exploited by considering sophisticated techniques ranging
from advanced feed forward architecture with residual connections [14] to convolutional approaches [15, 16] and

–
Recurrent Neural Networks [17,18] along with their many variants such as Echo-state Network [18 20], Long-Short

–
Term Memory [18,21 23] and Gated Recurrent Unit [18,24]. Moreover, some hybrid architectures have also been
proposed aiming to capture the temporal dependencies in the data with recurrent networks while performing a more
general feature extraction operation with convolutional layers [25,26].


Different reviews address the load forecasting topic by means of (not necessarily deep) neural networks. In [36]
the authors focus on the use of some deep learning architectures for load forecasting. However, this review lacks a
comprehensive comparative study of performance verified on common load forecasting benchmarks. The absence of
valid cost-performance metric does not allow the report to make conclusive statements. In [18] an exhaustive overview
of recurrent neural networks for short term load forecasting is presented. The very detailed work considers one layer
(not deep) recurrent networks only. A comprehensive summary of the most relevant researches dealing with STLF
employing recurrent neural networks, convolutional neural networks and seq2seq models is presented in Table 1. It
emerges that most of the works have been performed on different datasets, making it rather difficult - if not impossible to asses their absolute performance and, consequently, recommend the best state-of-the-art solutions for load forecast.


In this survey we consider the most relevant -and recent- deep architectures and contrast them in terms of performance
accuracy on open-source benchmarks. The considered architectures include recurrent neural networks, sequence
to sequence models and temporal convolutional neural networks. The experimental comparison is performed on
two different real-world datasets which are representatives of two distinct scenarios. The first one considers power
consumption at an individual household level with a signal characterized by high frequency components while the
second one takes into account aggregation of several consumers. Our contributions consist in:


2


A PREPRINT


_•_ A comprehensive review. The survey provides a comprehensive investigation of deep learning architectures
known to the smart grid literature as well as novel recent ones suitable for electric load forecasting.

_•_ A multi-step prediction strategy comparison for recurrent neural networks: we study and compare how different
prediction strategies can be applied to recurrent neural networks. To the best of our knowledge this work has
not been done yet for deep recurrent neural networks.

_•_ A relevant performance assessment. To the best of our knowledge, the present work provides the first systematic
experimental comparison of the most relevant deep learning architectures for the electric load forecasting
problems of individual and aggregated electric demand. It should be noted that envisaged architectures are
domain independent and, as such, can be applied in different forecasting scenarios.


The rest of this paper is organized as follows.
In Section 2 we formally introduce the forecasting problems along with the notation that will be used in this work. In
Section 3 we introduce Feed Forward Neural Networks (FNNs) and the main concepts relevant to the learning task. We
also provide a short review of the literature regarding the use of FNNs for the load forecasting problem.
In Section 4 we provide a general overview of Recurrent Neural Networks (RNNs) and their most advanced architectures:
Long Short-Term Memory and Gated Recurrent Unit networks.
In Section 5 Sequence To Sequence architectures (seq2seq) are discussed as a general improvement over recurrent
neural networks. We present both, simple and advanced models built on the sequence to sequence paradigm.
In Section 6 Convolutional Neural Networks are introduced and one of their most recent variant, the temporal
convolutional network (TCN), is presented as the state-of-the-art method for univariate time-series prediction.
In Section 7 the real-world datasets used for models comparison are presented. For each dataset, we provide a description
of the preprocessing operations and the techniques that have been used to validate the models performance.
Finally, In Section 8 we draw conclusions based on the performed assessments.


**2** **Problem Description**



Fixed-length Time Windows

(Sliding Window Apprach)


One fixed-length Time

Window







𝑡





Figure 1: A sliding windowed approach is used to frame the forecasting problem into a supervised machine learning
problem. The target signal **s** is split in multiple input output pairs ( **x** **t** _,_ **y** **t** ) _∀t ∈{n_ _T_ _, n_ _T_ + 1 _, . . ., T −_ _n_ _O_ _}_


In basic multi-step ahead electric load forecasting a univariate time series **s** = [ _s_ [0] _, s_ [1] _. . ., s_ [ _T_ ]] that spans through
several years is given. In this work, input data are presented to the different predictive families of models as a regressor
vector composed of fixed time-lagged data associated with a window size of length _n_ _T_ which slides over the time series.
Given this fixed length view of past values, a predictor _f_ aims at forecasting the next _n_ _O_ values of the time series.
In this work the forecasting problem is studied as a supervised learning problem. As such, given the input vector at
discrete time _t_ defined as **x** _t_ = [ _s_ [ _t −_ _n_ _T_ + 1] _, . . ., s_ [ _t_ ]] _∈_ IR _[n]_ _[T]_, the forecasting problem requires to infer the next _n_ _O_


3


A PREPRINT









|Notation|Description|
|---|---|
|_nT_|window size of the regressor vector|
|_nO_|time horizon of the forecast|
|_d −_1|number of exogenous variable|
|_u_|scalar value|
|**u**|vector/matrix|
|**u**~~T~~|vector/matrix transposed|
|_⊙_|elementwise product|
|*|convolution operator|
|_∗d_|dilated convolution operation<br>|
|_ℓ_|index for the_ l_~~_th_ ~~layer|
|**xt**|regressor vector at discrete time_ t_<br>(reference system: time-series)<br>**xt**_ ∈_IR_nT_ or** xt**_ ∈_IR_nT ×d_|
|**yt**|true output for the input sequence at time_ t_<br>(reference system: time-series)<br>**yt**_ ∈_IR_nO_|
|**ˆyt**|predicted output for the input sequence at time_ t_<br>(reference system: time-series)<br>**ˆyt**_ ∈_IR_nO_|
|**x**[_t_]|input vector of load & other features at time_ t_<br>(reference system: time window)<br>**x**[_t_]_ ∈_IR or** x**[_t_]_ ∈_IR_d_|
|_y_[_t_]|value of the load time-series at time_ t_ + 1<br>(reference system: time window)<br>_y_[_t_]_ ∈_IR|
|**zt**|exogenous features vector at time_ t_<br>(reference system: time series)<br>**zt**_ ∈_IR_nT ×d−_1|
|**z**[_t_]|exogenous features vector at time_ t_<br>(reference system:time window).<br>**z**[_t_]_ ∈_IR_d−_1|
|**h**|hidden state vector|
|**Θ**|model’s vector of parameters|
|_nH_|number of hidden neurons|


Table 2: The nomenclature used in this work.


measurements **y** **t** = [ _s_ [ _t_ + 1] _, . . ., s_ [ _t_ + _n_ _O_ ]] _∈_ IR _[n]_ _[O]_ or a subset of. To ease the notation we express the input and
output vectors in the reference system of the time window instead of the time series one. By following this approach,
the input vector at discrete time _t_ becomes **x** _t_ = [ _x_ _t_ [0] _, . . ., x_ _t_ [ _n_ _T_ _−_ 1]] _∈_ IR _[n]_ _[T]_ _, x_ _t_ [ _i_ ] = _s_ [ _i_ + 1 + _t −_ _n_ _T_ ] and the
corresponding output vector is **y** **t** = [ _y_ _t_ [ _n_ _T_ _−_ 1] _, . . ., y_ _t_ [ _n_ _T_ + _n_ _O_ _−_ 2]] _∈_ IR _[n]_ _[O]_ . _y_ _t_ characterizes the real output values
defined as _y_ _t_ [ _t_ ] = _x_ _t_ [ _t_ + 1] _,_ _∀t ∈_ _T_ . Similarly, we denote as **ˆy** **t** = _f_ ( **x** **t** ; **Θ** **[ˆ]** ) _∈_ IR _[n]_ _[O]_, the prediction vector provided
by a predictive model _f_ whose parameters vector **Θ** has been estimated by optimizing a performance function.


Without loss of generality, in the remaining of the paper, we drop the subscript _t_ from the inner elements of **x** **t** and **y** **t** .
The introduced notation, along with the sliding window approach, is depicted in Figure 1.


In certain applications we will additionally be provided with _d −_ 1 exogenous variables (e.g., the temperatures) each
of which representing a univariate time series aligned in time with the data of electricity demand. In this scenario
the components of the regressor vector become vectors, i.e., **x** **t** = [ **x** [0] _, . . .,_ **x** [ _n_ _T_ _−_ 1]] _∈_ IR _[n]_ _[T]_ _[ ×][d]_ . Indeed, each
element of the input sequence is represented as **x** [ _t_ ] = [ _x_ [ _t_ ] _, z_ 0 [ _t_ ] _, . . ., z_ _d−_ 2 [ _t_ ]] _∈_ IR _[d]_ where _x_ [ _t_ ] _∈_ IR is the scalar load
measurement at time _t_, while _z_ _k_ [ _t_ ] _∈_ IR is the scalar value of the _k_ _[th]_ exogenous feature.


The nomenclature used in this work is given in Table 2.


4


A PREPRINT


**3** **Feed Forward Neural Networks**


Feed Forward Neural Networks (FNNs) are parametric model families characterized by the universal function approximation property [37]. Their computational architectures are composed of a layered structure consisting of three
main building blocks: the input layer, the hidden layer(s) and the output layer. The number of hidden layers ( _L >_ 1 ),
determines the depth of the network, while the size of each layer, i.e., the number _n_ _H,ℓ_ of hidden units of the _ℓ_ _−_ th layer
defines its complexity in terms of neurons. FNNs provide only direct forward connections between two consecutive
layers, each connection associated with a trainable parameter; note that given the feedfoward nature of the computation
no recursive feedback is allowed. More in detail, given a vector **x** _∈_ IR _[n]_ _[T]_ fed at the network input, the FNN’s
computation can be expressed as:


**a** _ℓ_ = **W** _ℓ_ [T] **[h]** _[ℓ][−]_ **[1]** [+] **[ b]** _[ℓ]_ _[,]_ _ℓ_ = 1 _, ...L_ (1)
**h** _ℓ_ = _φ_ _ℓ_ ( **a** _ℓ_ ) (2)


where **h** 0 = **x** **t** _∈_ IR _[n]_ _[T]_ and **ˆy** **t** = **h** _L_ _∈_ IR _[n]_ _[O]_ .


Each layer _ℓ_ is characterized with its own parameters matrix **W** _ℓ_ _∈_ IR _[n]_ _[H,ℓ][−]_ [1] _[×][n]_ _[H,ℓ]_ and bias vector **b** _ℓ_ _∈_ IR _[n]_ _[H,ℓ]_ .
Hereafter, in order to ease the notation, we incorporate the bias term in the weight matrix, i.e., **W** _ℓ_ = [ **W** _ℓ_ ; **b** _ℓ_ ] and
**h** _ℓ_ = [ **h** _ℓ_ ; 1]. **Θ** = [ **W** 1 _, . . .,_ **W** _L_ ] groups all the network’s parameters.


Given a training set of _N_ input-output vectors in the ( **x** **i**, **y** **i** ) form, _i_ = 1 _, . . ., N_, the learning procedure aims at
identifying a suitable configuration of parameters **Θ** **[ˆ]** that minimizes a loss function _L_ evaluating the discrepancy
between the estimated values _f_ ( **x** **t** ; **Θ** ) and the measurements **y** **t** :


**ˆΘ** = arg min _.L_ ( **Θ** )
**Θ**


The mean squared error:



_L_ ( **Θ** ) = [1]

_N_



_N_
�( **y** **t** _−_ _f_ ( **x** **t** ; **Θ** )) [2] (3)


_t_ =1



is a very popular loss function for time series prediction and, not rarely, a regularization penalty term is introduced to
prevent overfitting and improve the generalization capabilities of the model



_L_ ( **Θ** ) = [1]

_N_



_N_
�( **y** **t** _−_ _f_ ( **x** **t** ; **Θ** )) [2] + Ω( **Θ** ) _._ (4)


_t_ =1



The most used regularization scheme controlling model complexity is the L2 regularization Ω( **Θ** ) = _λ∥_ **Θ** _∥_ 2 [2] [, being] _[ λ]_ [ a]
suitable hyper-parameter controlling the regularization strength.


As Equation 4 is not convex, the solution cannot be obtained in a closed form with linear equation solvers or convex
optimization techniques. Parameters estimation (learning procedure) operates iteratively e.g., by leveraging on the
gradient descent approach:

**Θ** _k_ = **Θ** _k−_ 1 _−_ _η∇_ **Θ** _L_ ( **Θ** ) (5)
��� **Θ** = **Θ** _k−_ 1


where _η_ is the learning rate and _∇_ **Θ** _L_ ( **Θ** ) the gradient w.r.t. **Θ** . Stochastic Gradient Descent (SGD), RMSProp [38],
Adagrad [39], Adam [40] are popular learning procedures. The learning procedure yields estimate **Θ** **[ˆ]** = **Θ** _k_ associated
with the predictive model _f_ ( **x** **t** ; **Θ** **[ˆ]** ).


In our work, deep FNNs are the baseline model architectures.


In multi-step ahead prediction the output layer dimension coincides with the forecasting horizon _n_ _O_ _>_ 1 . The dimension
of the input vector depends also on the presence of exogenous variables; this aspect is further discussed in Section 7.


**3.1** **Related Work**


The use of Feed Forward Neural networks in short term load forecasting dates back to the 90s. Authors in [11] propose
a shallow neural network with a single hidden layer to provide a 24-hour forecast using both load and temperature
information. In [10] one day ahead forecast is implemented using two different prediction strategies: one network
provides all 24 forecast values in a single shot (MIMO strategy) while another single output network provides the
day-ahead prediction by recursively feedbacking its last value estimate (recurrent strategy). The recurrent strategy


5


A PREPRINT


shows to be more efficient in terms of both training time and forecasting accuracy. In [41] the authors present a feed
forward neural network to forecast electric loads on a weekly basis. The sparsely connected feed forward architecture
receives the load time-series, temperature readings, as well as the time and day of the week. It is shown that the extra
information improves the forecast accuracy compared to an ARIMA model trained on the same task. [12] presents one
of the first multi-layer FNN to forecast the hourly load of a power system.


A detailed review concerning applications of artificial neural networks in short-term load forecasting can be found in [3].
However, this survey dates back to the early 2000s, and does not discuss deep models. More recently, architectural
variants of feed forward neural networks have been used; for example, in [14] a ResNet [42] inspired model is used to
provide day ahead forecast by leveraging on a very deep architecture. The article shows a significant improvement on
aggregated load forecasting when compared to other (not-neural) regression models on different datasets.


**4** **Recurrent Neural networks**


In this section we overview recurrent neural networks, and, in particular the Elmann Net architecture [43], Long-Short
Term Memory [44] and Gated Recurrent Unit [45] networks. Afterwords, we introduce deep recurrent neural networks
and discuss different strategies to perform multi-step ahead forecasting. Finally, we present related work in short-term
load forecasting that leverages on recurrent networks.


**4.1** **Elmann RNNs (ERNN)**


Elmann Recurrent Neural Networks (ERNN) were proposed in [43] to generalize feedforward neural networks for
better handling ordered data sequences like time-series.


The reason behind the effectiveness of RNNs in dealing with sequences of data comes from their ability to learn a
compact representation of the input sequence **x** **t** by means of a recurrent function _f_ that implements the following
mapping:
**h** [ _t_ ] = _f_ ( **h** [ _t −_ 1] _,_ **x** [ _t_ ]; **Θ** ) (6)



�̂ � �̂[0]





� UNFOLDING � ~�~� �[0] **...** �[�-1 ~~]~~





UNFOLDING











Figure 2: (Left) A simple RNN with a single input. The black box represents the delay operator which leads to Equation
6. (Right) The network after unfolding. Note that the structure reminds that of a (deep) feed forward neural network
but, here, each layer is constrained to share the same weights. **h** _init_ is the initial state of the network which is usually

set to zero.


By expanding Equation 6 and given a sequence of inputs **x** **t** = [ **x** [0] _, . . .,_ **x** [ _n_ _T_ _−_ 1]], **x** [ _t_ ] _∈_ IR _[d]_ the computation
becomes:


**a** [ _t_ ] = **W** _[T]_ **h** [ _t −_ 1] + **U** _[T]_ **x** [ _t_ ] (7)
**h** [ _t_ ] = _φ_ ( **a** [ _t_ ]) (8)

**y** [ _t_ ] = _ψ_ ( **V** _[T]_ **h** [ _t_ ]) (9)


where **W** _∈_ IR _[n]_ _[H]_ _[×][n]_ _[H]_, **U** _∈_ IR _[d][×][n]_ _[H]_, **V** _∈_ IR _[n]_ _[H]_ _[×][n]_ _[O]_ are the weight matrices for hidden-hidden, input-hidden,
hidden-output connections respectively, _φ_ ( _·_ ) is an activation function (generally the hyperbolic tangent one) and _ψ_ ( _·_ ) is
normally a linear function. The computation of a single module in an Elmann recurrent neural network is depicted in
Figure 3.


6


A PREPRINT



















Figure 3: A simple ERNN block with one cell implementing Equation 78 once rewritten as matrix concatenation:
**a** [ _t_ ] = [ **W** _,_ **U** ] _[T]_ [ **h** [ _t −_ 1] _,_ **x** [ _t_ ]], **h** [ _t_ ] = _φ_ ( **a** [ _t_ ] ), with [ **W** _,_ **U** ] _∈_ IR [(] _[n]_ _[H]_ [+] _[d]_ [)] _[×][n]_ _[H]_ and [ **h** [ _t −_ 1] _,_ **x** [ _t_ ]] _∈_ IR _[n]_ _[H]_ [+] _[d]_, Usually

_·_
_φ_ ( ) is the hyperbolic tangent.


It can be noted that an ERNN processes one element of the sequence at a time, preserving its inherent temporal order.
After reading an element from the input sequence **x** [ _t_ ] _∈_ IR _[d]_ the network updates its internal state **h** [ _t_ ] _∈_ IR _[n]_ _[H]_ using
both (a transformation of) the latest state **h** [ _t_ _−_ 1] and (a transformation of) the current input (Equation 6). The described
process can be better visualized as an acyclic graph obtained from the original cyclic graph (left side of Figure 2) via an
operation known as time unfolding (right side of Figure 2). It is of fundamental importance to point out that all nodes in
the unfolded network share the same parameters, as they are just replicas distributed over time.


The parameters of the network **Θ** = [ **W** _,_ **U** _,_ **V** ] are usually learned via Backpropagation Through Time (BPTT) [46,47],
a generalized version of standard Backpropagation. In order to apply gradient-based optimization, the recurrent neural
network has to be transformed through the unfolding procedure shown in Figure 2. In this way, the network is converted
into a FNN having as many layers as time intervals in the input sequence, and each layer is constrained to have the
same weight matrices. In practice Truncated Backpropagation Through Time [48] TBPTT( _τ_ _b_, _τ_ _f_ ) is used. The method
processes an input window of length _n_ _T_ one timestep at a time and runs BPTT for _τ_ _b_ timesteps every _τ_ _f_ steps. Notice
that having _τ_ _b_ _< n_ _T_ does not limit the memory capacity of the network as the hidden state incorporates information
taken from the whole sequence. Despite that, setting _τ_ _b_ to a very low number may result in poor performance. In the
literature BPTT is considered equivalent to TBPTT( _τ_ _b_ = _n_ _T_, _τ_ _f_ = 1 ). In this work we used epoch-wise Truncated
BPTT i.e., TBPTT( _τ_ _b_ = _n_ _T_, _τ_ _f_ = _n_ _T_ ) to indicate that the weights update is performed once a whole sequence has
been processed.


Despite of the model simplicity, Elmann RNNs are hard to train due to ineffectiveness of gradient (back)propagation.
In fact, it emerges that the propagation of gradient is effective for short-term connections but is very likely to fail for
long-term ones, when the gradient norm usually shrinks to zero or diverges. These two behaviours are known as the
vanishing gradient and the exploding gradient problems [49,50] and were extensively studied in the machine learning
community.


**4.2** **Long Short-Term Memory (LSTM)**


Recurrent neural networks with Long Short-Term Memory (LSTM) were introduced to cope with the vanishing and
exploding gradients problems occurring in ERNNs and, more in general, in standard RNNs [44]. LSTM networks
maintain the same topological structure of ERNN but differ in the composition of the inner module - or cell.


7


A PREPRINT





























Figure 4: Long-Short Term Memory block with one cell.


Each LSTM cell has the same input and output as an ordinary ERNN cell but, internally, it implements a gated system
that controls the neural information processing (see Figure Figure 3 and 4). The key feature of gated networks is their
ability to control the gradient flow by acting on the gate values; this allows to tackle the vanishing gradient problem, as
LSTM can maintain its internal memory unaltered for long time intervals. Notice from the equations below that the
inner state of the network results as a linear combination of the old state and the new state (Equation 14). Part of the
old state is preserved and flows forward while in the ERNN the state value is completely replaced at each timestep
(Equation 8). In detail, the neural computation is:


**i** [ **t** ] = _ψ_ ( **W** **f** **h** [ _t −_ 1] + **U** **f** **x** [ _t_ ]) (10)
**f** [ **t** ] = _ψ_ ( **W** **i** **h** [ _t −_ 1] + **U** **i** **x** [ _t_ ]) (11)
**o** [ **t** ] = _ψ_ ( **W** **o** **h** [ _t −_ 1] + **U** **o** **x** [ _t_ ]) (12)
� **c** [ **t** ] = _φ_ ( **W** **c** **h** [ _t −_ 1] + **U** **c** **x** [ _t_ ]) (13)
**c** [ **t** ] = **f** [ _t_ ] _⊙_ **c** [ _t −_ 1] + **i** [ _t_ ] _⊙_ � **c** [ _t_ ] (14)
**h** [ **t** ] = **o** [ _t_ ] _⊙_ _φ_ ( **c** [ _t_ ]) (15)


where **W** **f** _,_ **W** **i** _,_ **W** **o** _,_ **W** **c** _∈_ IR _[n]_ _[H]_ _[×][n]_ _[H]_, **U** **f** _,_ **U** **i** _,_ **U** **o** _,_ **U** **c** _∈_ IR _[n]_ _[H]_ _[×][d]_ are parameters to be learned, _⊙_ is the Hadamard

_·_ _·_
product, _ψ_ ( ) is generally a sigmoid activation while _φ_ ( ) can be any non-linear one (hyperbolic tangent in the original
paper). The cell state **c** [ _t_ ] encodes the - so far learned - information from the input sequence. At timestep _t_ the flow
of information within the unit is controlled by three elements called gates: the forget gate **f** [ _t_ ] controls the cell state’s
content and changes it when obsolete, the input gate **i** [ _t_ ] controls which state value will be updated and how much, � **c** [ _t_ ],
finally the output gate **o** [ _t_ ] produces a filtered version of the cell state and serves it as the network’s output **h** [ _t_ ] [51].


**4.3** **Gated Recurrent Units (GRU)**


Firstly introduced in [45], GRUs are a simplified variant of LSTM and, as such, belong to the family of gated RNNs.
GRUs distinguish themselves from LSTMs for merging in one gate functionalities controlled by the forget gate and the
input gate. This kind of cell ends up having just two gates, which results in a more parsimonious architecture compared
to LSTM that, instead, has three gates.


8


A PREPRINT





























Figure 5: Gated Recurrent Unit memory block with one cell.


The basic components of a GRU cell are outlined in Figure 5, whereas the neural computation is controlled by:


**u** [ **t** ] = _ψ_ ( **W** **u** **h** [ _t −_ 1] + **U** **u** **x** [ _t_ ]) (16)
**r** [ **t** ] = _ψ_ ( **W** **r** **h** [ _t −_ 1] + **U** **r** **x** [ _t_ ]) (17)
**h** �[ **t** ] = _φ_ ( **W** **c** ( **r** [ _t_ ] _⊙_ **h** [ _t −_ 1]) + **U** **c** **x** [ _t_ ]) (18)

**h** [ **t** ] = **u** [ **t** ] _⊙_ **h** [ _t −_ 1] + (1 _−_ **u** [ _t_ ]) _⊙_ **h** [�] [ _t_ ] (19)


where **W** **u** _,_ **W** **r** _,_ **W** **c** _∈_ IR _[n]_ _[H]_ _[×][n]_ _[H]_, **U** **u** _,_ **U** **r** _,_ **U** **c** _∈_ IR _[n]_ _[H]_ _[×][d]_ are the parameters to be learned, _ψ_ ( _·_ ) is generally a sigmoid
activation while _φ_ ( _·_ ) can be any kind of non-linearity (in the original work it was an hyperbolic tangent). **u** [ _t_ ] and **r** [ _t_ ]
are the update and the reset gates, respectively. Several works in the natural language processing community show that
GRUs perform comparably to LSTM but train generally faster due to the lighter computation [52,53].


**4.4** **Deep Recurrent Neural Networks**


All recurrent architectures presented so far are characterized by a single layer. In turn, this implies that the computation
is composed by an affine transformation followed by a non-linearity. That said, the concept of depth in RNN is less
straightforward than in feed-forward architectures. Indeed, the later ones become deep when the input is processed by a
large number of non-linear transformations before generating the output values. However, according to this definition,
an unfolded RNN is already a deep model given its multiple non-linear processing layers. That said, a deep multi-level
processing can be applied to all the transition functions (input-hidden, hidden-hidden, hidden-output) as there are no
intermediate layers involved in these computations [54]. Deepness can also be introduced in recurrent neural networks
by stacking recurrent layers one on top of the other [55]. As this deep architecture is more intriguing, in this work, we
refer it as a Deep RNN. By iterating the RNN computation, the function implemented by the deep architecture can be
represented as:
**h** _ℓ_ [ _t_ ] = _f_ ( **h** _ℓ_ [ _t −_ 1] _,_ **h** _ℓ−_ 1 [ _t_ ]; **Θ** ) _ℓ_ = 1 _,_ 2 _, ..., L_ (20)
where **h** _ℓ_ [ _t_ ] is the hidden state at timestep _t_ for layer _ℓ_ . Notice that **h** 0 [ _t_ ] = **x** [ _t_ ] . It has been empirically shown in
several works that Deep RNNs are better to capture the temporal hierarchy exhibited by time-series then their shallow
counterpart [54,56,57]. Of course, hybrid architectures having different layers -recurrent or not- can be considered as
well.


**4.5** **Multi-Step Prediction Schemes**


There are five different architecture-independent strategies for multi-step ahead forecasting [58]:


9


A PREPRINT


**Recursive strategy (Rec)** a single model is trained to perform a one-step ahead forecast given the input sequence.
Subsequently, during the operational phase, the forecasted output is recursively fedback and considered to be the correct
one. By iterating _n_ _O_ times this procedure we generate the forecast values at time _t_ + _n_ _O_ . The procedure is described in
Algorithm 1, where **x** [1 :] is the input vector without its first element while the _vectorize_ ( _·_ ) procedure concatenates the
scalar output _y_ to the exogenous input variables.


**Algorithm 1** Recursive Strategy (Rec) for Multi-Step Forecasting


1: **x** _←_ **x** **t**
2: **o** _←_ empty list
3: _k ←_ 1
4: **while** _k < n_ _O_ + 1 **do**
5: _o ←_ _f_ ( **x** )
6: **o** _←_ _concatenate_ ( **o** _, o_ )
7: **x** _←_ _concatenate_ ( **x** [1 :] _, vectorize_ ( _o_ )))
8: _k ←_ _k_ + 1
9: **end while**
10: **return o** as **ˆy** **t**


To summarize, the predictor _f_ receives in input a vector **x** of length _n_ _T_ and outputs a scalar value _o_ .


**Direct strategy** design a set of _n_ _O_ independent predictors _f_ _k_ _, k_ = 1 _, . . ., n_ _O_, each of which providing a forecast at
time _t_ + _k_ . Similarly to the recursive strategy, each predictor _f_ _k_ outputs a scalar value _o_, but the input vector is the
same to all the predictors. Algorithm 2 details the procedure.


**Algorithm 2** Direct Strategy for Multi-Step Forecasting


1: **x** _←_ **x** **t**
2: **o** _←_ empty list
3: _k ←_ 1
4: **while** _k < n_ _O_ + 1 **do**
5: **o** _←_ _concatenate_ ( **o** _, f_ _k_ ( **x** ))
6: _k ←_ _k_ + 1
7: **end while**
8: **return o** as **ˆy** **t**


**DirRec strategy** [59] is a combination of the above two strategies. Similar to the direct approach, _n_ _O_ models are
used, but here, each predictor leverages on an enlarged input set, obtained by adding the results of the forecast at the
previous timestep. The procedure is detailed in Algorithm 3.


**Algorithm 3** DirRec Strategy for Multi-Step Forecasting


1: **x** _←_ **x** **t**
2: **o** _←_ empty list
3: _k ←_ 1
4: **while** _k < n_ _O_ + 1 **do**
5: _o ←_ _f_ _k_ ( **x** )
6: **o** _←_ _concatenate_ ( **o** _, o_ )
7: **x** _←_ _concatenate_ ( **x** _, vectorize_ ( _o_ ))
8: _k ←_ _k_ + 1
9: **end while**
10: **return o** as **ˆy** **t**


**MIMO strategy** (Multiple input - Multiple output) [60], a single predictor _f_ is trained to forecast a whole output
sequence of length _n_ _O_ in one-shot, i.e., differently from the previous cases the output of the model is not a scalar but a

vector:


**ˆy** **t** = _f_ ( **x** **t** )


10


A PREPRINT


**DIRMO strategy** [61], represents a trade-off between the Direct strategy and the MIMO strategy. It divides the _n_ _O_
steps forecasts into smaller forecasting problems, each of which of length _s_ . It follows that _⌈_ _[n]_ _s_ _[O]_ _[⌉]_ [predictors are used to]

solve the problem.


Given the considerable computational demand required by RNNs during training, we focus on multi-step forecasting
strategies that are computationally cheaper, specifically, Recursive and MIMO strategies [58]. We will call them
RNN-Rec and RNN-MIMO.


Given the hidden state **h** [ _t_ ] at timestep _t_, the hidden-output mapping is obtained through a fully connected layer on top
of the recurrent neural network. The objective of this dense network is to learn the mapping between the last state of the
recurrent network, which represents a kind of lossy summary of the task-relevant aspect of the input sequence, and
the output domain. This holds for all the presented recurrent networks and is consistent with Equation 9. In this work
RNN-Rec and RNN-MIMO differ in the cardinality of the output domain, which is 1 for the former and _n_ _O_ for the
latter, meaning that in Equation 9 either **V** _∈_ IR _[n]_ _[H]_ _[×]_ [1] or **V** _∈_ IR _[n]_ _[H]_ _[×][n]_ _[O]_ . The objective function is:



_L_ ( **Θ** ) = [1]

_n_ _O_



_n_ _O_ _−_ 1


ˆ

� ( _y_ [ _t_ ] _−_ _y_ [ _t_ ]) [2] + Ω( **Θ** ) (21)


_t_ =0



**4.6** **Related work**


In [17] an Elmann recurrent neural network is considered to provide hourly load forecasts. The study also compares
the performance of the network when additional weather information such as temperature and humidity are fed to the
model. The authors conclude that, as expected, the recurrent network benefits from multi-input data and, in particular,
weather ones. [28] makes use of ERNN to forecast household electric consumption obtained from a suburban area
in the neighbours of Palermo (Italy). In addition to the historical load measurements, the authors introduce several
features to enhance the model’s predictive capabilities. Besides the weather and the calendar information, a specific
ad-hoc index was created to assess the influence of the use of air-conditioning equipment on the electricity demand. In
recent years, LSTMs have been adopted in short term load forecasting, proving to be more effective then traditional
time-series analysis methods. In [21] LSTM is shown to outperform traditional forecasting methods being able to
exploit the long term dependencies in the time series to forecast the day-ahead load consumption. Several works
proved to be successful in enhancing the recurrent neural network capabilities by employing multivariate input data.
In [22] the authors propose a deep, LSTM based architecture that uses past measurements of the whole household
consumption along with some measurements from selected appliances to forecast the consumption of the subsequent
time interval (i.e., a one step prediction). In [23] a LSTM-based network is trained using a multivariate input which
includes temperature, holiday/working day information, date and time information. Similarly, in [31] a power demand
forecasting model based on LSTM shows an accuracy improvement compared to more traditional machine learning
techniques such as Gradient Boosting Trees and Support Vector Regression.


GRUs have not been used much in the literature as LSTM networks are often preferred. That said, the use of GRU-based
networks is reported in [18], while a more recent study [24] uses GRUs for the daily consumption forecast of individual
customers. Thus, investigating deep GRU-based architectures is a relevant scientific topic, also thanks to their faster
convergence and simpler structure compared to LSTM [52].


Despite all these promising results, an extensive study of recurrent neural networks [18], and in particular of ERNN,
LSTM, GRU, ESN [62] and NARX, concludes that none of the investigated recurrent architectures manages to
outperform the others in all considered experiments. Moreover, the authors noticed that recurrent cells with gated
mechanisms like LSTM and GRU perform comparably well to much simpler ERNN. This may indicate that in short-term
load forecasting gating mechanism may be unnecessary; this issue is further investigated -and evidence found- in the
present work.


**5** **Sequence To Sequence models**


Sequence To Sequence (seq2seq) architectures [63] or encoder-decoder models [45] were initially designed to solve
RNNs inability to produce output sequences of arbitrary length. The architecture was firstly used in neural machine

–
translation [45,64,65] but has emerged as the golden standard in different fields such as speech recognition [66 68] and
image captioning [69].


The core idea of this general framework is to employ two networks resulting in an encoder-decoder architecture. The
first neural network (possibly deep) _f_, an encoder, reads the input sequence **x** **t** _∈_ IR _[n]_ _[T]_ _[ ×][d]_ of length _n_ _T_ one timestep
at a time; the computation generates a, generally lossy, fixed dimensional vector representation of it **c** = _f_ ( **x** **t** _,_ **Θ** _f_ ),


11


A PREPRINT











DECODER


Figure 6: seq2seq (Encoder-Decoder) architecture with a general Recurrent Neural network both for the encoder and
the decoder. Assuming a Teacher Forcing training process, the solid lines in the decoder represents the training phase
while the dotted lines depicts the values’path during prediction.


**c** _∈_ IR _[d]_ _[′]_ . This embedded representation is usually called context in the literature and can be the last hidden state of
the encoder or a function of it. Then, a second neural network _g_ - the decoder - will learn how to produce the output
sequence **ˆy** **t** _∈_ IR _[n]_ _[O]_ given the context vector, i.e., **ˆy** = _g_ ( **c** _,_ **Θ** _g_ ) . The schematics of the whole architecture is depicted
in Figure 6.


The encoder and the decoder modules are generally two recurrent neural networks trained end-to-end to minimize the
objective function:



_L_ ( **Θ** ) =



_n_ _O_ _−_ 1


ˆ

� ( _y_ [ _t_ ] _−_ _y_ [ _t_ ]) [2] + Ω( **Θ** ) _,_ **Θ** = [ **Θ** _f_ _,_ **Θ** _g_ ] (22)


_t_ =0



ˆ
_y_ [ _t_ ] = _g_ ( _y_ [ _t −_ 1] _,_ **h** [ _t −_ 1] _,_ **c** ; **Θ** ) (23)

where ˆ _y_ [ _t_ ] is the decoder’s estimate at time _t_, _y_ [ _t_ ] is the real measurement, **h** [ _t −_ 1] is the decoder’s last state, **c** is the
context vector from the encoder, **x** is the input sequence and Ω( **Θ** ) the regularization term. The training procedure
for this type of architecture is called teacher forcing [70]. As shown in Figure 6 and explained in Equation 23, during
training, the decoder’s input at time _t_ is the ground-truth value _y_ [ _t −_ 1], which is then used to generate the next state
**h** [ _t_ ] and, then, the estimate ˆ _y_ [ _t_ ]. During inference the true values are unavailable and replaced by the estimates:


ˆ
_y_ [ _t_ ] = _g_ (ˆ _y_ [ _t −_ 1] _,_ **h** [ _t −_ 1] _,_ **c** ; **Θ** ) _._ (24)


This discrepancy between training and testing results in errors accumulating over time during inference. In the literature
this problem is often referred to as exposure bias [71]. Several solutions have been proposed to address this problem;
in [72] the authors present scheduled sampling, a curriculum learning strategy that gradually changes the training
process by switching the decoder’s inputs from ground-truth values to model’s predictions. The _professor forcing_
algorithm, introduced in [73], uses an adversarial framework to encourage the dynamics of the recurrent network to be
the same both at training and operational (test) time. Finally, in recent years, reinforcement learning methods have been
adopted to train sequence to sequence models; a comprehensive review is presented in [74].


In this work we investigate two sequence to sequence architectures, one trained via _teacher forcing_ (TF) and one using
_self-generated_ (SG) samples. The former is characterized by Equation 23 during training while Equation 24 is used
during prediction. The latter architecture adopts Equation 24 both for training and prediction. The decoder’s dynamics
are summarized in Figure 7. It is clear that the two training procedures differ in the decoder’s input source: ground-truth
values in teacher forcing, estimated values in self-generated training.


**5.1** **Related Work**


Only recently seq2seq models have been adopted in short term load forecasting. In [33] a LSTM based encoderdecoder model is shown to produce superior performance compared to standard LSTM. In [75] the authors introduce
an adaptation of RNN based sequence-to-sequence architectures for time-series forecasting of electrical loads to


12


|... ... ...|Col2|
|---|---|
|RNN<br>RNN<br>RNN<br>Ã[]<br>Ã[+1]<br>Ã[+]<br>...|RNN<br>RNN<br>RNN<br>Ã[]<br>Ã[+1]<br>Ã[+]<br>...|
|||



DECODER (TF)



A PREPRINT

|̂ [] ̂ [+1] ... ̂ [+]<br>...<br>RNN RNN RNN<br>... ... ...<br>...<br>RNN RNN RNN<br>...<br>Ã[]|Col2|
|---|---|
|||



DECODER (SG)



Figure 7: (Left) decoder with ground-truth inputs (Teacher Forcing). (Right) Decoder with self-generated inputs.


demonstrate its better performance with respect to a suite of models ranging from standard RNNs to classical time
series techniques.


**6** **Convolutional Neural Networks**


Convolutional Neural Networks (CNNs) [76] are a family of neural networks designed to work with data that can be
structured in a grid-like topology. CNNs were originally used on two dimensional and three-dimensional images, but
they are also suitable for one-dimensional data such as univariate time-series. Once recognized as a very efficient

–
solution for image recognition and classification [42,77 79], CNNs have experienced wide adoption in many different

–
computer vision tasks [80 84]. Moreover, sequence modeling tasks, like short term electric load forecasting, have been
mainly addressed with recurrent neural networks, but recent research indicates that convolutional networks can also
attain state-of-the-art-performance in several applications including audio generation [85], machine translation [86] and
time-series prediction [87].


As the name suggests, these kind of networks are based on a discrete convolution operator that produces an output
feature map **f** by sliding a kernel **w** over the input **x** . Each element in the output feature map is obtained by summing
up the result of the element-wise multiplication between the input patch (i.e., a slice of the input having the same
dimensionality of the kernel) and the kernel. The number of kernels (filters) _M_ used in a convolutional layer determines
the depth of the output volume (i.e., the number of output feature maps). To control the other spatial dimensions of
the output feature maps two hyper-parameters are used: stride and padding. Stride represents the distance between
two consecutive input patches and can be defined for each direction of motion. Padding refers to the possibility of
implicitly enlarging the inputs by adding (usually) zeros at the borders to control the output size w.r.t the input one.
Indeed, without padding, the dimensionality of the output would be reduced after each convolutional layer.


Considering a 1D time-series **x** _∈_ IR _[n]_ _[T]_ and a one-dimensional kernel **w** _∈_ IR _[k]_, the _i_ _[th]_ element of the convolution
between **x** and **w** is:



_f_ ( _i_ ) = ( **x** _∗_ **w** )( _i_ ) =



_k−_ 1
� _x_ ( _i −_ _j_ ) _w_ ( _j_ ) (25)

_j_ =0



with **f** _∈_ IR _[n]_ _[T]_ _[ −][k]_ [+1] if no zero-padding is used, otherwise padding matches the input dimensionality, i.e., **f** _∈_ IR _[n]_ _[T]_ .
Equation 25 is referred to the one-dimensional input case but can be easily extended to multi-dimensional inputs (e.g.,
images, where **x** _∈_ IR _[W][ ×][H][×][D]_ ) [88]. The reason behind the success of these networks can be summarized in the
following three points:


_•_ local connectivity: each hidden neuron is connected to a subset of input neurons that are close to each other
(according to specific spatio-temporal metric). This property allows the network to drastically reduce the
number of parameters to learn (w.r.t. a fully connected network) and facilitate computations.


_•_ parameter sharing: the weights used to compute the output neurons in a feature map are the same, so that the
same kernel is used for each location. This allows to reduce the number of parameters to learn.


13


Figure 8: A 3 layers CNN with causal convolution (no dilation), the receptive field _r_ is 4.



A PREPRINT


Figure 9: A 3 layers CNN with dilated causal convolutions. The
dilation factor _d_ grows on each layer by a factor of two and the
kernel size _k_ is 2, thus the output neuron is influence by 8 input
neurons, i.e., the history size is 8




_•_ translation equivariance: the network is robust to an eventual shifting of its input.


In our work we focus on a convolutional architecture inspired by Wavenet [85], a fully probabilistic and autoregressive
model used for generating raw audio wave-forms and extended to time-series prediction tasks [87]. Up to the authors’
knowledge this architecture has never been proposed to forecast the electric load. A recent empirical comparison between
temporal convolutional networks and recurrent networks has been carried out in [89] on tasks such as polymorphic
music and charter-sequence level modelling. The authors were the first to use the name Temporal Convolutional
Networks (TCNs) to indicate convolutional networks which are autoregressive, able to process sequences of arbitrary
length and output a sequence of the same length. To achieve the above the network has to employ causal (dilated)
convolutions and residual connections should be used to handle a very long history size.


**Dilated Causal Convolution (DCC)** Being TCNs a family of autoregressive models, the estimated value at time
_t_ must depend only on past samples and not on future ones (Figure 9). To achieve this behavior in a Convolutional
Neural Network the standard convolution operator is replaced by causal convolution. Moreover, zero-padding of length
(filter size - 1) is added to ensure that each layer has the same length of the input layer. To further enhance the network
capabilities _dilated causal convolutions_ are used, allowing to increase the receptive field of the network (i.e., the number
of input neurons to which the filter is applied) and its ability to learn long-term dependencies in the time-series. Given a
one-dimensional input **x** _∈_ IR _[n]_ _[T]_, and a kernel **w** _∈_ IR _[k]_, a dilated convolution output using a dilation factor _d_ becomes:



_f_ ( _i_ ) = ( **x** _∗_ _d_ **w** )( _i_ ) =



_k−_ 1
� _x_ ( _i −_ _dj_ ) _w_ ( _j_ ) (26)

_j_ =0



This is a major advantage w.r.t simple causal convolutions, as in the later case the receptive field _r_ grows linearly with
the depth of the network _r_ = _k_ ( _L −_ 1) while with dilated convolutions the dependence is exponential _r_ = 2 _[L][−]_ [1] _k_,
ensuring that a much larger history size is used by the network.


**Residual Connections** Despite the implementation of dilated convolution, the CNN still needs a large number of
layers to learn the dynamics of the inputs. Moreover, performance often degrade with the increase of the network depth.
The degradation problem has been first addressed in [42] where the authors propose a deep residual learning framework.
The authors observe that for a _L_ -layers network with a training error _ϵ_, inserting _k_ extra layers on top of it should either
leave the error unchanged or improve it. Indeed, in the worst case scenario, the new _k_ stacked non linear layers should
learn the identity mapping **y** = _H_ ( **x** ) = **x** where **x** is the output of the network having _L_ layers and **y** is the output of
the network with _L_ + _k_ layers. Although almost trivial, in practice, neural networks experience problems in learning
this identity mapping. The proposed solution suggests these stacked layers to fit a residual mapping _F_ ( **x** ) = _H_ ( **x** ) _−_ **x**
instead of the desired one, _H_ ( **x** ) . The original mapping is recast into _F_ ( **x** ) + **x** which is realized by feed forward
neural networks with shortcut connections; in this way the identity mapping is learned by simply driving the weights of
the stacked layers to zero.


By means of the two aforementioned principles, the temporal convolutional network is able to exploit a large history size
in an efficient manner. Indeed, as observed in [89], these models present several computational advantages compared to
RNNs. In fact, they have lower memory requirements during training and the predictions for later timesteps are not
done sequentially but can be computed in parallel exploiting parameter sharing. Moreover, TCNs training is much


14


A PREPRINT








|Conv1D|Col2|Dropout|Col4|ReLU|
|---|---|---|---|---|
|Conv1D|||||











Figure 10: TCN Architecture. **x** **t** is the vector of historical loads along with the exogenous features for the time
window indexes from 0 to _n_ _T_, **z** **t** is the vector of exogenous variables related to the last _n_ _O_ indexes of the time window
(when available), **ˆy** **t** is the output vector. Residual Blocks are composed by a 1D Dilated Causal Convolution, a ReLU
activation and Dropout. The square box represents a concatenation between (transformed) exogenous features and
(transformed) historical loads.


more stable than that involving RNNs allowing to avoid the exploding/vanishing gradient problem. For all the above,
TCNs have demonstrated to be promising area of research for time series prediction problems and here, we aim to
assess their forecasting performance w.r.t state-of-the-art models in short-term load forecasting. The architecture used
in our work is depicted in Figure 10, which is, except for some minor modifications, the network structure detailed
in [87]. In the first layer of the network we process separately the load information and, when available, the exogenous
information such as temperature readings. Later the results will be concatenated together and processed by a deep
residual network with _L_ layers. Each layer consists of a residual block with 1D dilated causal convolution, a rectified
linear unit (ReLU) activation and finally dropout to prevent overfitting. The output layer consists of 1x1 convolution
which allows the network to output a one-dimensional vector **y** _∈_ IR _[n]_ _[T]_ having the same dimensionality of the input
vector **x** . To approach multi-step forecasting, we adopt a MIMO strategy.


**6.1** **Related Work**


In the short-term load forecasting relevant literature, CNNs have not been studied to a large extent. Indeed, until
recently, these models were not considered for any time-series related problem. Still, several works tried to address the
topic; in [15] a deep convolutional neural network model named DeepEnergy is presented. The proposed network is
inspired by the first architectures used in ImageNet challenge (e.g, [77]), alternating convolutional and pooling layers,
halving the width of the feature map after each step. According to the provided experimental results, DeepEnergy can
precisely predict energy load in the next three days outperforming five other machine learning algorithms including
LSTM and FNN. In [16] a CNN is compared to recurrent and feed forward approaches showing promising results on a
benchmark dataset. In [25] a hybrid approach involving both convolutional and recurrent architectures is presented. The
authors integrate different input sources and use convolutional layers to extract meaningful features from the historic
load while the recurrent network main task is to learn the system’s dynamics. The model is evaluated on a large dataset
containing hourly loads from a city in North China and is compared with a three-layer feed forward neural network. A
different hybrid approach is presented in [26], the authors process the load information in parallel with a CNN and
an LSTM. The features generated by the two networks are then used as an input for a final prediction network (fully
connected) in charge of forecasting the day-ahead load.


**7** **Performance Assessment**


In this section we perform evaluation and assessment of all the presented architectures. The testing is carried out by
means of three use cases that are based on two different datasets used as benchmarks. We first introduce the performance
metrics that we considered for both network optimization and testing, then describe the datasets that have been used and
finally we discuss results.


15


A PREPRINT


**7.1** **Performance Metrics**


The efficiency of the considered architectures has been measured and quantified using widely adopted error metrics.
Specifically, we adopted the Root mean squared error (RMSE) and the Mean Absolute Error (MAE):



_n_ _O_ _−_ 1
� (ˆ _y_ _i_ [ _t_ ] _−_ _y_ _i_ [ _t_ ]) [2]


_t_ =0



RMSE =



~~�~~
~~�~~
�


_N_

� [1]



_N_ _−_ 1
�


_i_ =0



1

_n_ _O_



_n_ _O_ _−_ 1
� _|_ ˆ _y_ _i_ [ _t_ ] _−_ _y_ _i_ [ _t_ ] _|_


_t_ =0



MAE = [1]

_N_



_N_ _−_ 1
�


_i_ =0



1

_n_ _O_



where _N_ is the number of input-output pairs provided to the model in the course of testing, _y_ _i_ [ _t_ ] and ˆ _y_ _i_ [ _t_ ] are respectively
the real load values and the estimated load values at time _t_ for sample _i_ (i.e., the _i −_ _th_ time window). _⟨·⟩_ is the mean
operator, _∥· ∥_ 2 is the euclidean L2 norm, while _∥· ∥_ 1 is the L1 norm. **y** _∈_ IR _[n]_ _[O]_ and **ˆy** _∈_ IR _[n]_ _[O]_ are the real load
values and the estimated load values for one sample, respectively. Still, a more intuitive and indicative interpretation of
prediction efficiency of the estimators can be expressed by the normalized root mean squared error which, differently
from the two above metrics, is independent from the scale of the data:


RMSE
NRMSE % = _·_ 100
_y_ _max_ _−_ _y_ _min_


where _y_ _max_ and _y_ _min_ are the maximum and minimum value of training dataset, respectively. In order to quantify the
proportion of variance in the target that is explained by the forecasting methods we consider also the R [2] index:



TSS [)]



R [2] = [1]

_N_



_N_ _−_ 1
�



� (1 _−_ [RSS] TSS

_i_ =0



_n_ _O_



RSS = [1]

_n_ _O_



_n_ _O_
�



(ˆ _y_ _i_ [ _t_ ] _−_ _y_ _i_ [ _t_ ]) [2] TSS = [1]

_n_

_t_



_n_ _O_


¯

�( _y_ _i_ [ _t_ ] _−_ _y_ _i_ ) [2]


_t_



where ¯ _y_ _i_ = _n_ 1 _O_



_n_ _O_
� _y_ _i_ [ _t_ ]

_t_



All considered models have been implemented in Keras 2.12 [90] with Tensorflow [91] as backend. The experiments
are executed on a Linux cluster with an Intel(R) Xeon(R) Silver CPU and an Nvidia Titan XP.


**7.2** **Use Case I**


The first use case considers the Individual household electric power consumption data set (IHEPC) which contains
2.07M measurements of electric power consumption for a single house located in Sceaux (7km of Paris, France).
Measurements are collected every minute between December 2006 and November 2010 (47 months) [29]. In this study
we focus on prediction of the "Global active power" parameter. Nearly 1.25% of measurements are missing, still, all
the available ones come with timestamps. We reconstruct the missing values using the mean power consumption for
the corresponding time slot across the different years of measurements. In order to have a unified approach we have
decided to resample the dataset using a sampling rate of 15 minutes which is a widely adopted standard in modern
smart meters technologies. In Table 3 the sample size are outlined for each dataset.


In this use case we performed the forecasting using only historical load values. The right side of Figure 11 depicts the
average weekly electric consumption. As expected, it can be observed that the highest consumption is registered in the
morning and evening periods of day when the occupancy of resident houses is high. Moreover, the average load profile
over a week clearly shows that weekdays are similar while weekends present a different trend of consumption.


The figure shows that the data are characterized by high variance. The prediction task consists in forecasting the electric
load for the next day, i.e., 96 timesteps ahead.


In order to assess the performance of the architectures we hold out a portion of the data which denotes our test set and
comprises the last year of measurements. The remaining measurements are repeatedly divided in two sets, keeping
aside a month of data every five ones. This process allows us to build a training set and a validation set for which
different hyper-parameters configurations can be evaluated. Only the best performing configuration is later evaluated on
the test set.


16


A PREPRINT





Dataset **Train** **Test**


IHEPC 103301 35040

GEFCom2014 44640 8928

Table 3: Sample size of train, validation and test sets for each dataset.






|4.0|Col2|Col3|Col4|
|---|---|---|---|
|2.0<br>2.5<br>3.0<br>3.5<br>4.0<br> Power (MW)||||
|2.0<br>2.5<br>3.0<br>3.5<br>4.0<br> Power (MW)||||
|2.0<br>2.5<br>3.0<br>3.5<br>4.0<br> Power (MW)||||
|2.0<br>2.5<br>3.0<br>3.5<br>4.0<br> Power (MW)||||
|~~Mon~~<br>~~Tue~~<br>~~Wed~~<br>~~Thu~~<br>~~Fri~~<br>Time<br>0.0<br>0.5<br>1.0<br>1.5<br>Active||||
|~~Mon~~<br>~~Tue~~<br>~~Wed~~<br>~~Thu~~<br>~~Fri~~<br>Time<br>0.0<br>0.5<br>1.0<br>1.5<br>Active||||
|~~Mon~~<br>~~Tue~~<br>~~Wed~~<br>~~Thu~~<br>~~Fri~~<br>Time<br>0.0<br>0.5<br>1.0<br>1.5<br>Active||||
|~~Mon~~<br>~~Tue~~<br>~~Wed~~<br>~~Thu~~<br>~~Fri~~<br>Time<br>0.0<br>0.5<br>1.0<br>1.5<br>Active|||~~Sat~~<br>~~Sun~~|


|220<br>200<br>180<br>(MW)<br>160<br>140 Load<br>120<br>100<br>80<br>Mon Tue Wed Thu<br>Ti|Col2|Col3|
|---|---|---|
|~~Mon~~<br>~~Tue~~<br>~~Wed~~<br>~~Thu~~<br>T<br>80<br>100<br>120<br>140<br>160<br>180<br>200<br>220<br>Load (MW)|||
|~~Mon~~<br>~~Tue~~<br>~~Wed~~<br>~~Thu~~<br>T<br>80<br>100<br>120<br>140<br>160<br>180<br>200<br>220<br>Load (MW)|||
|~~Mon~~<br>~~Tue~~<br>~~Wed~~<br>~~Thu~~<br>T<br>80<br>100<br>120<br>140<br>160<br>180<br>200<br>220<br>Load (MW)|||
|~~Mon~~<br>~~Tue~~<br>~~Wed~~<br>~~Thu~~<br>T<br>80<br>100<br>120<br>140<br>160<br>180<br>200<br>220<br>Load (MW)|||
|~~Mon~~<br>~~Tue~~<br>~~Wed~~<br>~~Thu~~<br>T<br>80<br>100<br>120<br>140<br>160<br>180<br>200<br>220<br>Load (MW)|||
|~~Mon~~<br>~~Tue~~<br>~~Wed~~<br>~~Thu~~<br>T<br>80<br>100<br>120<br>140<br>160<br>180<br>200<br>220<br>Load (MW)|||
|~~Mon~~<br>~~Tue~~<br>~~Wed~~<br>~~Thu~~<br>T<br>80<br>100<br>120<br>140<br>160<br>180<br>200<br>220<br>Load (MW)||~~Fri~~<br>~~Sat~~<br>~~Sun~~<br>me|



Figure 11: Weekly statistics for the electric load in the whole IHEPC(Left) and GEFCom2014datasets (right). The bold
line is the mean curve, the dotted line is the median and the green area covers one standard deviation from the mean.


**7.3** **Use Case II and III**


The other two use cases are based on the GEFCom2014dataset [35], which was made available for an online forecasting
competition that lasted between August 2015 and December 2015. The dataset contains 60.6k hourly measurements of
(aggregated) electric power consumption collected by ISO New England between January 2005 and December 2011.
Differently from the IHEPCdataset, temperature values are also available and are used by the different architectures to
enhance their prediction performance. In particular the input variables being used for forecasting the subsequent _n_ _O_
at timestep _t_ include: several previous load measurements, the temperature measurements for the previous timesteps
registered by 25 different stations, hour, day, month and year of the measurements. We apply standard normalization
to load and temperature measurements while for the other variables we simply apply one-hot encoding, i.e., a _K_ dimensional vector in which one of the elements equals 1, and all remaining elements equal 0 [92]. On the right side
of Figure 11 we observe the average load and the data dispersion on a weekly basis. Compared to IHEPC, the load
profiles look much more regular. This meets intuitive expectations as the load measurements in the first dataset come
from a single household, thus the randomness introduced by user behaviour makes more remarkable impact on the
results. On the opposite, the load information in GEFCom2014comes from the aggregation of the data provided by
several different smart meters; clustered data exhibits a more stable and regular pattern. The main task of these use
cases, as well the previous one, consists in forecasting the electric load for the next day, i.e., 24 timesteps ahead. The
hyper-parameters optimization and the final score for the models follow the same guidelines provided for IHEPC, the
number of points for each subset is described in Table 3.


**7.4** **Results**


The compared architectures are the ones presented in previous sections with one exception. We have additionally
considered a deeper variant of a feed forward neural network with residual connections which is named DFNN in the
remainder of the work. In accordance to the findings of [93] we have employed a 2-shortcut network, i.e., the input
undergoes two affine transformations each followed by a non linearity before being summed to its original values. For
regularization purposes we have included Dropout and Batch Normalization [94] in each residual block. We have
additionally inserted this model in the results comparison because it represents an evolution of standard feed forward
neural networks which is expected to better handle highly complex time-series data.


Table 4 summarizes the best configurations found trough grid search for each model and use case. For both datasets we
experimented different input sequences of length _n_ _T_ . Finally, we used a window size of four days, which represents the
best trade-off between performance and memory requirements. The output sequence length _n_ _O_ is fixed to one day. For
each model we identified the optimal number of stacked layers in the network _L_, the number of hidden units per layer
_n_ _H_, the regularization coefficient _λ_ (L2 regularization) and the dropout rate _p_ _d_ . Moreover, for TCN we additionally
tuned the width _k_ of the convolutional kernel and the number of filters applied at each layer _M_ (i.e., the depth of each


17


A PREPRINT


**ERNN** **LSTM** **GRU** **seq2seq**
**Hyperparameters** **Dataset** **FNN** **DFNN TCN**
**Rec MIMO Rec MIMO Rec MIMO TF** **SG**


**IHPEC** 3 6 8 3 1 2 1 2 1 1 1

L **GEFCOM** 1 6 6 4 4 4 2 4 1 2 1

**GEFCOM** _exog_ 1 6 8 2 1 4 1 2 2 2 3


**IHPEC** 256-128 _x_ 2 50 10 30 20 20 10 50 30 50
_n_ _H_ **GEFCOM** 60 30 20 50 15 20 30 20 10 50
**GEFCOM** _exog_ 60 30 10 30 30 50 10 15 20 20-15-10


**IHPEC** 0.001 0.0005 0.005 0.001 0.001 0.001 0.001 0.001 0.0005 0.01 0.01

_λ_ **GEFCOM** 0.01 0.0005 0.01 0.01 0.0005 0.001 0.001 0.01 0.0005 0.01 0.01

**GEFCOM** _exog_ 0.005 0.0005 0.005 0.0005 0.001 0.0005 0.0005 0.001 0.01 0.001 0.01


**IHPEC** 0.1 0.1 0.1 0.0 0.0 0.0 0.0 0.0 0.0 0.1 0.2

_p_ _drop_ **GEFCOM** 0.1 0.1 0.1 0.1 0.0 0.1 0.0 0.1 0.0 0.1 0.1
**GEFCOM** _exog_ 0.1 0.1 0.1 0.0 0.0 0.0 0.0 0.1 0.0 0.1 0.0


**IHPEC** 2, 32
_k,_ M **GEFCOM** 2, 16
**GEFCOM** _exog_ 2, 64


Table 4: Best configurations found via Grid Search for the IHEPCdataset, GEFCom2014dataset and GEFCom2014with
exogenous features.


output volume after the convolution operation). The dilation factor is increased exponentially with the depth of the
network, i.e. _d_ = 2 _[ℓ]_ with _ℓ_ being the _ℓ_ _−_ _th_ layer of the network.


Table 5 summarizes the test scores of the presented architectures obtained for the IHEPCdataset. Certain similarities
among networks trained for different uses cases can be spotted out already at this stage. In particular, we observe that
all models exploit a small number of neurons. This is not usual in deep learning but - at least for recurrent architectures

- is consistent with [18]. With some exceptions, recurrent networks benefit from a less strict regularization; dropout is
almost always set to zero and _λ_ values are small.


Among Recurrent Neural Networks we observe that, in general, the MIMO strategy outperforms the recursive one in
this multi step prediction task. This is reasonable in such a scenario. Indeed, the recursive strategy, differently from the
MIMO one, is highly sensitive to errors accumulation which, in a highly volatile time series as the one addressed here,
results in a very inaccurate forecast. Among the MIMO models we observe that gated networks perform significantly
better than simple Elmann network one. This suggests that gated systems are effectively learning to better exploit the
temporal dependency in the data. In general we notice that all the models, except the RNNs trained with recursive
strategy, achieve comparable performance and none really stands out. It is interesting to comment that GRU-MIMO and
LSTM-MIMO outperform sequence to sequence architectures which are supposed to better model complex temporal
dynamics like the one exhibited by the residential load curve. Nevertheless, by observing the performance of recurrent
networks trained with the recursive strategy, this behaviour is less surprising. In fact, compared with the aggregated
load profiles, the load curve belonging to a single smart meter is way more volatile and sensitive to customers behaviour.
For this reason, leveraging geographical and socio-economic features that characterize the area where the user lives
may allow deep networks to generate better predictions.


For visualization purposes we compare all the models performance on a single day prediction scenario on the left side
of Figure 12. On the right side of Figure 12 we quantify the differences between the best predictor (the GRU-MIMO)
and the actual measurements; the thinner the line the closer the prediction to the true data. Furthermore, in this Figure,
we concatenate multiple day predictions to have a wider time span and evaluate the model predictive capabilities. We
observe that the model is able to generate a prediction that correctly models the general trend of the load curve but fails
to predict steep peaks. This might come from the design choice of using MSE as the optimization metric, which could
discourage deep models to predict high peaks as large errors are hugely penalized, and therefore, predicting a lower and
smoother function results in better performance according to this metric. Alternatively, some of the peaks may simply
represent noise due to particular user behaviour and thus unpredictable by definition.


18


A PREPRINT


|3.5<br>3.0<br>2.5 kW)|Meas<br>gru_<br>dfnn<br>tcn<br>lstm_<br>rnn_<br>lstm_<br>rnn_r<br>gru_r<br>fnn|
|---|---|
|1.5<br>2.0<br>ve Power (|1.5<br>2.0<br>ve Power (|





Figure 12: (Right) Predictive performance of all the models on a single day for IHEPCdataset. The left portion of the
image shows (part of) the measurements used as input while the right side with multiple lines represents the different
predictions. (Left) Difference between the best model’s predictions (GRU-MIMO) and the actual measurements. The
thinner the line the closer the prediction is to the true data.


**RMSE** **MAE** **NRMSE** R [2]


**FNN** 0 _._ 76 _±_ 0 _._ 01 0 _._ 53 _±_ 0 _._ 01 10 _._ 02 _±_ 0 _._ 17 0 _._ 250 _±_ 0 _._ 026

**DFNN** 0 _._ 75 _±_ 0 _._ 01 0 _._ 53 _±_ 0 _._ 01 9 _._ 90 _±_ 0 _._ 05 0 _._ 269 _±_ 0 _._ 007

**TCN** 0 _._ 76 _±_ 0 _._ 008 0 _._ 54 _±_ 0 _._ 00 10 _._ 07 _±_ 0 _._ 11 0 _._ 245 _±_ 0 _._ 017


**MIMO** 0 _._ 79 _±_ 0 _._ 00 0 _._ 56 _±_ 0 _._ 00 10 _._ 33 _±_ 0 _._ 08 0 _._ 201 _±_ 0 _._ 012
**ERNN**
**Rec** 0 _._ 88 _±_ 0 _._ 02 0 _._ 69 _±_ 0 _._ 03 11 _._ 61 _±_ 0 _._ 29 0 _._ 001 _±_ 0 _._ 039


**MIMO** 0 _._ 75 _±_ 0 _._ 00 0 _._ 53 _±_ 0 _._ 00 9 _._ 85 _±_ 0 _._ 04 0 _._ 276 _±_ 0 _._ 006
**LSTM**
**Rec** 0 _._ 84 _±_ 0 _._ 06 0 _._ 60 _±_ 0 _._ 07 11 _._ 06 _±_ 0 _._ 74 0 _._ 085 _±_ 0 _._ 125



**MIMO 0** _**.**_ **75** _**±**_ **0** _**.**_ **00** **0** _**.**_ **52** _**±**_ **0** _**.**_ **00** **9** _**.**_ **83** _**±**_ **0** _**.**_ **03** **0** _**.**_ **279** _**±**_ **0** _**.**_ **004**
**GRU**

**Rec** 0 _._ 89 _±_ 0 _._ 02 0 _._ 70 _±_ 0 _._ 02 11 _._ 64 _±_ 0 _._ 23 0 _._ 00 _±_ 0 _._ 04



**TF** 0 _._ 78 _±_ 0 _._ 01 0 _._ 57 _±_ 0 _._ 02 10 _._ 22 _±_ 0 _._ 17 0 _._ 221 _±_ 0 _._ 026
**seq2seq**
**SG** 0 _._ 76 _±_ 0 _._ 01 0 _._ 53 _±_ 0 _._ 01 10 _._ 00 _±_ 0 _._ 14 0 _._ 253 _±_ 0 _._ 03

Table 5: Individual household electric power consumption data set results. Each model’s mean score ( _±_ one standard.
deviation) comes from 10 repeated training processes.


The load curve of the second dataset (GEFCom2014) results from the aggregation of several different load profiles
producing a smoother load curve when compared with the individual load case. Hyper-parameters optimization and the
final score for the models can be found in Table 4.


Table 6 and Table 7 show the experimental results obtained by the models in two different scenarios. In the former case,
only load values were provided to the models while in the latter scenario the input vector has been augmented with the
exogenous features described before. Compared to the previous dataset this time series exhibits a much more regular
pattern; as such we expect the prediction task to be easier. Indeed, we can observe a major improvement in terms of
performance across all the models. As already noted in [22,95] the prediction accuracy increases significantly when the
forecasting task is carried out on a smooth load curve (resulting from the aggregation of many individual consumers).


We can observe that, in general, all models except plain FNNs benefit from the presence of exogenous variables.


When exogenous variables are adopted, we notice a major improvement by RNNs trained with the recursive strategy
which outperform MIMO ones. This increase in accuracy can be attributed to a better capacity of leveraging the
exogenous time series of temperatures to yield a better load forecast. Moreover, RNNs with MIMO strategy gain
negligible improvements compared to their performance when no extra-feature is provided. This kind of architectures
use a feedforward neural network to map their final hidden state to a sequence of _n_ _O_ values, i.e., the estimates.
Exogenous variables are elaborated directly by this FNN, which, as observed above, shows to have problems in handling
both load data and extra information. Consequently, a better way of injecting exogenous variables in MIMO recurrent
network needs to be found in order to provide a boost in prediction performance comparable to the one achieved by
employing the recursive strategy.


19


A PREPRINT








|275<br>250<br>225<br>200 (MW)|Col2|Col3|
|---|---|---|
|200<br>225<br>250<br>275<br> (MW)||~~150~~<br>~~200~~<br>~~25~~<br>1 Hour)|



Figure 13: (Right) Predictive performance of all the models on a single day for GEFCom2014dataset. The left portion
of the image shows (part of) the measurements used as input while the right side with multiple lines represents the
different predictions. (Left) Difference between the best model’s predictions (LSTM-Rec) and the actual measurements.
The thinner the line the closer the prediction to the true data.


**RMSE** **MAE** **NRMSE** R [2]


**FNN** 21 _._ 1 _±_ 2 _._ 5 15 _._ 5 _±_ 2 _._ 1 7 _._ 01 _±_ 0 _._ 82 0 _._ 833 _±_ 0 _._ 041

**DFNN** 22 _._ 4 _±_ 6 _._ 2 17 _._ 1 _±_ 6 _._ 2 7 _._ 44 _±_ 2 _._ 01 0 _._ 801 _±_ 0 _._ 124

**TCN** 17 _._ 2 _±_ 0 _._ 1 11 _._ 5 _±_ 0 _._ 1 5 _._ 71 _±_ 0 _._ 14 0 _._ 891 _±_ 0 _._ 00


**MIMO** 18 _._ 0 _±_ 0 _._ 3 11 _._ 9 _±_ 0 _._ 4 5 _._ 99 _±_ 0 _._ 11 0 _._ 879 _±_ 0 _._ 046
**ERNN**
**Rec** 27 _._ 0 _±_ 2 _._ 3 20 _._ 7 _±_ 2 _._ 5 8 _._ 95 _±_ 0 _._ 78 0 _._ 732 _±_ 0 _._ 046


**MIMO** 19 _._ 5 _±_ 0 _._ 5 13 _._ 7 _±_ 0 _._ 6 6 _._ 47 _±_ 0 _._ 18 0 _._ 861 _±_ 0 _._ 007
**LSTM**
**Rec** 25 _._ 6 _±_ 2 _._ 2 18 _._ 4 _±_ 1 _._ 3 8 _._ 52 _±_ 0 _._ 72 0 _._ 757 _±_ 0 _._ 041


**MIMO** 19 _._ 0 _±_ 0 _._ 2 13 _._ 1 _±_ 0 _._ 3 6 _._ 29 _±_ 0 _._ 07 0 _._ 868 _±_ 0 _._ 003
**GRU**
**Rec** 26 _._ 7 _±_ 3 _._ 3 19 _._ 8 _±_ 3 _._ 1 8 _._ 85 _±_ 1 _._ 09 0 _._ 737 _±_ 0 _._ 064


**TF** 21 _._ 5 _±_ 2 _._ 1 15 _._ 4 _±_ 1 _._ 9 7 _._ 13 _±_ 0 _._ 69 0 _._ 829 _±_ 0 _._ 034
**seq2seq**
**SG** **17** _**.**_ **1** _**±**_ **0** _**.**_ **2** **11** _**.**_ **3** _**±**_ **0** _**.**_ **2** **5** _**.**_ **67** _**±**_ **0** _**.**_ **06** **0** _**.**_ **893** _**±**_ **0** _**.**_ **002**

Table 6: GEFCom2014results without any exogenous variable. Each model’s mean score ( _±_ one standard. deviation)
comes from 10 repeated training processes.


For reasons that are similar to those discussed above, sequence to sequence models trained via _teacher forcing_ (seq2seqTF) experienced an improvement when exogenous features are used. Still, seq2seq trained in free-running mode
(seq2seq-SG) proves to be a valid alternative to standard seq2seq-TF producing high quality predictions in all use cases.
The absence of a discrepancy between training and inference in terms of data generating distribution shows to be an
advantage as seq2seq-SG is less sensitive to noise and error propagation.


Finally, we notice that TCNs perform well in all the presented use cases. Considering their lower memory requirements
in the training process along with their inherent parallelism this type of networks represents a promising alternative to
recurrent neural networks for short-term load forecasting.


The results of predictions are presented in the same fashion as for the previous use case in Figure 13. Observe that,
in general, all the considered models are able to produce reasonable estimates as sudden picks in consumption are
smoothed. Therefore, predictors greatly improve their accuracy when predicting day ahead values for the aggregated
load curves with respect to individual households scenario.


**8** **Conclusions**


In this work we have surveyed and experimentally evaluated the most relevant deep learning models applied to the
short-term load forecasting problem, paving the way for standardized assessment and identification of the most optimal
solutions in this field. The focus has been given to the three main families of models, namely, Recurrent Neural
Networks, Sequence to Sequence Architectures and recently developed Temporal Convolutional Neural Networks.


20


A PREPRINT


**RMSE** **MAE** **NRMSE** R [2]


**FNN** 27 _._ 9 _±_ 2 _._ 8 20 _._ 8 _±_ 2 _._ 4 9 _._ 28 _±_ 0 _._ 93 0 _._ 709 _±_ 0 _._ 062

**DFNN** 23 _._ 0 _±_ 1 _._ 2 15 _._ 6 _±_ 0 _._ 7 7 _._ 62 _±_ 0 _._ 41 0 _._ 805 _±_ 0 _._ 021

**TCN** 15 _._ 4 _±_ 1 _._ 5 8 _._ 6 _±_ 1 _._ 7 5 _._ 00 _±_ 0 _._ 22 0 _._ 917 _±_ 0 _._ 007


**MIMO** 17 _._ 9 _±_ 0 _._ 3 11 _._ 7 _±_ 0 _._ 3 5 _._ 94 _±_ 0 _._ 01 0 _._ 883 _±_ 0 _._ 004
**ERNN**
**Rec** 14 _._ 7 _±_ 1 _._ 0 8 _._ 6 _±_ 1 _._ 0 4 _._ 88 _±_ 0 _._ 19 0 _._ 925 _±_ 0 _._ 005


**MIMO** 18 _._ 1 _±_ 1 _._ 3 12 _._ 1 _±_ 1 _._ 3 6 _._ 01 _±_ 0 _._ 42 0 _._ 877 _±_ 0 _._ 018
**LSTM**
**Rec** **13** _**.**_ **8** _**±**_ **0** _**.**_ **6** **7** _**.**_ **5** _**±**_ **0** _**.**_ **3** **4** _**.**_ **59** _**±**_ **0** _**.**_ **18** **0** _**.**_ **930** _**±**_ **0** _**.**_ **006**


**MIMO** 17 _._ 8 _±_ 0 _._ 2 11 _._ 7 _±_ 0 _._ 2 5 _._ 93 _±_ 0 _._ 07 0 _._ 882 _±_ 0 _._ 002
**GRU**
**Rec** 16 _._ 7 _±_ 0 _._ 5 10 _._ 0 _±_ 0 _._ 6 5 _._ 54 _±_ 0 _._ 15 0 _._ 898 _±_ 0 _._ 006


**TF** 14 _._ 3 _±_ 1 _._ 0 8 _._ 5 _±_ 0 _._ 9 4 _._ 74 _±_ 0 _._ 32 0 _._ 924 _±_ 0 _._ 014
**seq2seq**
**SG** 15 _._ 9 _±_ 1 _._ 8 9 _._ 8 _±_ 1 _._ 8 5 _._ 28 _±_ 0 _._ 60 0 _._ 907 _±_ 0 _._ 021

Table 7: GEFCom2014results with exogenous variables. Each model’s mean score ( _±_ one standard. deviation) comes
from 10 repeated training processes.


An architectural description along with a technical discussion on how multi-step ahead forecasting is achieved, has
been provided for each considered model. Moreover, different forecasting strategies are discussed and evaluated,
identifying advantages and drawbacks for each of them. The evaluation has been carried out on the three real-world use
cases that refer to two distinct scenarios for load forecasting. Indeed, one use case deals with dataset coming from a
single household while the other two tackle the prediction of a load curve that represents several aggregated meters,
dispersed over the wide area. Our findings concerning application of recurrent neural networks to short-term load
forecasting, show that the simple ERNN performs comparably to gated networks such as GRU and LSTM when adopted
in aggregated load forecasting. Thus, the less costly alternative provided by ERNN may represent the most effective
solution in this scenario as it allows to reduce the training time without remarkable impact on prediction accuracy. On
the contrary, a significant difference exists for single house electric load forecasting where the gated networks shows to
be superior to Elmann ones suggesting that the gated mechanism allows to better handle irregular time series. Sequence
to Sequence models have demonstrated to be quite efficient in load forecasting tasks even though they seem to fail in
outperforming RNNs. In general we can claim that seq2seq architectures do not represent a golden standard in load
forecasting as they are in other domains like natural language processing. In addition to that, regarding this family of
architectures, we have observed that teacher forcing may not represent the best solution for training seq2seq models on
short-term load forecasting tasks. Despite being harder in terms of convergence, free-running models learn to handle
their own errors, avoiding the discrepancy between training and testing that is a well known issue for teacher forcing. It
turns out to be worth efforts to further investigate capabilities of seq2seq models trained with intermediate solutions
such as _professor forcing_ . Finally, we evaluated the recently developed Temporal Convolutional Neural Networks which
demonstrated convincing performance when applied to load forecasting tasks. Therefore, we strongly believe that the
adoption of these networks for sequence modelling in the considered field is very promising and might even introduce a
significant advance in this area that is emerging as a key importance for future Smart Grid developments.


**Acknowledgment**


This project is carried out within the frame of the Swiss Centre for Competence in Energy Research on the Future Swiss
Electrical Infrastructure (SCCER-FURIES) with the financial support of the Swiss Innovation Agency (Innosuisse SCCER program).


**References**


[1] X. Fang, S. Misra, G. Xue, and D. Yang. Smart grid — the new and improved power grid: A survey. _IEEE_
_Communications Surveys Tutorials_, 14(4):944–980, Fourth 2012.


[2] Eisa Almeshaiei and Hassan Soltan. A methodology for electric power load forecasting. _Alexandria Engineering_
_Journal_, 50(2):137 – 144, 2011.


[3] H. S. Hippert, C. E. Pedreira, and R. C. Souza. Neural networks for short-term load forecasting: a review and
evaluation. _IEEE Transactions on Power Systems_, 16(1):44–55, Feb 2001.


21


A PREPRINT


[4] Jiann-Fuh Chen, Wei-Ming Wang, and Chao-Ming Huang. Analysis of an adaptive time-series autoregressive
moving-average (arma) model for short-term load forecasting. _Electric Power Systems Research_, 34(3):187–196,
1995.


[5] Shyh-Jier Huang and Kuang-Rong Shih. Short-term load forecasting via arma model identification including
non-gaussian process considerations. _IEEE Transactions on power systems_, 18(2):673–679, 2003.


[6] Martin T Hagan and Suzanne M Behr. The time series approach to short term load forecasting. _IEEE Transactions_
_on Power Systems_, 2(3):785–791, 1987.


[7] Chao-Ming Huang, Chi-Jen Huang, and Ming-Li Wang. A particle swarm optimization to identifying the armax
model for short-term load forecasting. _IEEE Transactions on Power Systems_, 20(2):1126–1133, 2005.


[8] Hong-Tzer Yang, Chao-Ming Huang, and Ching-Lien Huang. Identification of armax model for short term load
forecasting: An evolutionary programming approach. In _Power Industry Computer Application Conference, 1995._
_Conference Proceedings., 1995 IEEE_, pages 325–330. IEEE, 1995.


[9] Guy R Newsham and Benjamin J Birt. Building-level occupancy data to improve arima-based electricity use
forecasts. In _Proceedings of the 2nd ACM workshop on embedded sensing systems for energy-efficiency in building_,
pages 13–18. ACM, 2010.


[10] K. Y. Lee, Y. T. Cha, and J. H. Park. Short-term load forecasting using an artificial neural network. _IEEE_
_Transactions on Power Systems_, 7(1):124–132, Feb 1992.


[11] D. C. Park, M. A. El-Sharkawi, R. J. Marks, L. E. Atlas, and M. J. Damborg. Electric load forecasting using an
artificial neural network. _IEEE Transactions on Power Systems_, 6(2):442–449, May 1991.


[12] Dipti Srinivasan, A.C. Liew, and C.S. Chang. A neural network short-term load forecaster. _Electric Power Systems_
_Research_, 28(3):227 – 234, 1994.


[13] I. Drezga and S. Rahman. Short-term load forecasting with local ann predictors. _IEEE Transactions on Power_
_Systems_, 14(3):844–850, Aug 1999.


[14] K. Chen, K. Chen, Q. Wang, Z. He, J. Hu, and J. He. Short-term load forecasting with deep residual networks.
_IEEE Transactions on Smart Grid_, pages 1–1, 2018.


[15] Ping-Huan Kuo and Chiou-Jye Huang. A high precision artificial neural networks model for short-term energy
load forecasting. _Energies_, 11(1), 2018.


[16] K. Amarasinghe, D. L. Marino, and M. Manic. Deep neural networks for energy load forecasting. In _2017 IEEE_
_26th International Symposium on Industrial Electronics (ISIE)_, pages 1483–1488, June 2017.


[17] Siddarameshwara Nayaka, Anup Yelamali, and Kshitiz Byahatti. Electricity short term load forecasting using
elman recurrent neural network. pages 351 – 354, 11 2010.


[18] Filippo Maria Bianchi, Enrico Maiorino, Michael C. Kampffmeyer, Antonello Rizzi, and Robert Jenssen.
An overview and comparative analysis of recurrent neural networks for short term load forecasting. _CoRR_,
abs/1705.04378, 2017.


[19] Filippo Maria Bianchi, Enrico De Santis, Antonello Rizzi, and Alireza Sadeghian. Short-term electric load
forecasting using echo state networks and pca decomposition. _IEEE Access_, 3:1931–1943, 2015.


[20] Elena Mocanu, Phuong H Nguyen, Madeleine Gibescu, and Wil L Kling. Deep learning for estimating building
energy consumption. _Sustainable Energy, Grids and Networks_, 6:91–99, 2016.


[21] Jian Zheng, Cencen Xu, Ziang Zhang, and Xiaohua Li. Electric load forecasting in smart grids using long-shortterm-memory based recurrent neural network. In _Information Sciences and Systems (CISS), 2017 51st Annual_
_Conference on_, pages 1–6. IEEE, 2017.


[22] Weicong Kong, Zhao Yang Dong, Youwei Jia, David J Hill, Yan Xu, and Yuan Zhang. Short-term residential load
forecasting based on lstm recurrent neural network. _IEEE Transactions on Smart Grid_, 2017.


[23] Salah Bouktif, Ali Fiaz, Ali Ouni, and Mohamed Serhani. Optimal deep learning lstm model for electric
load forecasting using feature selection and genetic algorithm: Comparison with machine learning approaches.
_Energies_, 11(7):1636, 2018.


[24] Yixing Wang, Meiqin Liu, Zhejing Bao, and Senlin Zhang. Short-term load forecasting with multi-source data
using gated recurrent unit neural networks. _Energies_, 11:1138, 05 2018.


[25] Wan He. Load forecasting via deep neural networks. _Procedia Computer Science_, 122:308 – 314, 2017. 5th
International Conference on Information Technology and Quantitative Management, ITQM 2017.


22


A PREPRINT


[26] Chujie Tian, Jian Ma, Chunhong Zhang, and Panpan Zhan. A deep neural network model for short-term load
forecast based on long short-term memory network and convolutional neural network. _Energies_, 11:3493, 12
2018.

[27] Tao Hong, Pierre Pinson, and Shu Fan. Global energy forecasting competition 2012. _International Journal of_
_Forecasting_, 30(2):357 – 363, 2014.

[28] Antonino Marvuglia and Antonio Messineo. Using recurrent artificial neural networks to forecast household
electricity consumption. _Energy Procedia_, 14:45 – 55, 2012. 2011 2nd International Conference on Advances in
Energy Engineering (ICAEE).

[29] Dua Dheeru and Efi Karra Taniskidou. UCI machine learning repository, 2017.

[30] Smart grid, smart city, australian govern., australia, canberray.

[31] Yao Cheng, Chang Xu, Daisuke Mashima, Vrizlynn L. L. Thing, and Yongdong Wu. Powerlstm: Power demand
forecasting using long short-term memory neural network. In Gao Cong, Wen-Chih Peng, Wei Emma Zhang,
Chengliang Li, and Aixin Sun, editors, _Advanced Data Mining and Applications_, pages 727–740, Cham, 2017.
Springer International Publishing.

[32] Umass smart dataset. `[http://traces.cs.umass.edu/index.php/Smart/Smart](http://traces.cs.umass.edu/index.php/Smart/Smart)`, 2017.

[33] Daniel L Marino, Kasun Amarasinghe, and Milos Manic. Building energy load forecasting using deep neural
networks. In _Industrial Electronics Society, IECON 2016-42nd Annual Conference of the IEEE_, pages 7046–7051.
IEEE, 2016.

[34] Henning Wilms, Marco Cupelli, and Antonello Monti. Combining auto-regression with exogenous variables in
sequence-to-sequence recurrent neural networks for short-term load forecasting. In _2018 IEEE 16th International_
_Conference on Industrial Informatics (INDIN)_, pages 673–679. IEEE, 2018.

[35] Tao Hong, Pierre Pinson, Shu Fan, Hamidreza Zareipour, Alberto Troccoli, and Rob J. Hyndman. Probabilistic
energy forecasting: Global energy forecasting competition 2014 and beyond. _International Journal of Forecasting_,
32(3):896 – 913, 2016.

[36] A. Almalaq and G. Edwards. A review of deep learning methods applied on load forecasting. In _2017 16th IEEE_
_International Conference on Machine Learning and Applications (ICMLA)_, pages 511–516, Dec 2017.

[37] Balázs Csanád Csáji. Approximation with artificial neural networks.

[38] G. Hinton. `[http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)` .

[39] Matthew D. Zeiler. Adadelta: An adaptive learning rate method. 1212, 12 2012.

[40] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. _CoRR_, abs/1412.6980, 2014.

[41] S.T. Chen, D.C. Yu, and A.R. Moghaddamjo. Weather sensitive short-term load forecasting using nonfully
connected artificial neural network. _IEEE Transactions on Power Systems (Institute of Electrical and Electronics_
_Engineers); (United States)_, (3), 8 1992.

[42] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In _2016_
_IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2016, Las Vegas, NV, USA, June 27-30,_
_2016_, pages 770–778, 2016.

[43] Jeffrey L. Elman. Finding structure in time. _COGNITIVE SCIENCE_, 14(2):179–211, 1990.

[44] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. _Neural Computation_, 9(8):1735–1780, 1997.

[45] Kyunghyun Cho, Bart van Merrienboer, Çaglar Gülçehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk,
and Yoshua Bengio. Learning phrase representations using RNN encoder-decoder for statistical machine translation.
In _Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing, EMNLP 2014,_
_October 25-29, 2014, Doha, Qatar, A meeting of SIGDAT, a Special Interest Group of the ACL_, pages 1724–1734,
2014.

[46] Paul J. Werbos. Backpropagation through time: What it does and how to do it. 1990.

[47] D. E. Rumelhart, G. E. Hinton, and R. J. Williams. Parallel distributed processing: Explorations in the microstructure of cognition, vol. 1. chapter Learning Internal Representations by Error Propagation, pages 318–362. MIT
Press, Cambridge, MA, USA, 1986.

[48] Ronald J. Williams and Jing Peng. An efficient gradient-based algorithm for on-line training of recurrent network
trajectories. _Neural Computation_, 2, 09 1998.

[49] Y. Bengio, P. Simard, and P. Frasconi. Learning long-term dependencies with gradient descent is difficult. _Trans._
_Neur. Netw._, 5(2):157–166, March 1994.


23


A PREPRINT


[50] Razvan Pascanu, Tomas Mikolov, and Yoshua Bengio. On the difficulty of training recurrent neural networks. In
_Proceedings of the 30th International Conference on International Conference on Machine Learning - Volume 28_,
ICML’13, pages III–1310–III–1318. JMLR.org, 2013.

[51] K. Greff, R. K. Srivastava, J. Koutník, B. R. Steunebrink, and J. Schmidhuber. Lstm: A search space odyssey.
_IEEE Transactions on Neural Networks and Learning Systems_, 28(10):2222–2232, Oct 2017.

[52] Junyoung Chung, Çaglar Gülçehre, Kyunghyun Cho, and Yoshua Bengio. Empirical evaluation of gated recurrent
neural networks on sequence modeling. _CoRR_, abs/1412.3555, 2014.

[53] Wenpeng Yin, Katharina Kann, Mo Yu, and Hinrich Schütze. Comparative study of cnn and rnn for natural
language processing. _arXiv preprint arXiv:1702.01923_, 2017.

[54] Razvan Pascanu, Çaglar Gülçehre, Kyunghyun Cho, and Yoshua Bengio. How to construct deep recurrent neural
networks. _CoRR_, abs/1312.6026, 2013.

[55] Jürgen Schmidhuber. Learning complex, extended sequences using the principle of history compression. _Neural_
_Comput._, 4(2):234–242, March 1992.

[56] Alex Graves, Abdel-rahman Mohamed, and Geoffrey E. Hinton. Speech recognition with deep recurrent neural
networks. _CoRR_, abs/1303.5778, 2013.

[57] Michiel Hermans and Benjamin Schrauwen. Training and analysing deep recurrent neural networks. In C. J. C.
Burges, L. Bottou, M. Welling, Z. Ghahramani, and K. Q. Weinberger, editors, _Advances in Neural Information_
_Processing Systems 26_, pages 190–198. Curran Associates, Inc., 2013.

[58] Souhaib Ben Taieb, Gianluca Bontempi, Amir F. Atiya, and Antti Sorjamaa. A review and comparison of
strategies for multi-step ahead time series forecasting based on the nn5 forecasting competition. _Expert Systems_
_with Applications_, 39(8):7067 – 7083, 2012.

[59] Antti Sorjamaa and Amaury Lendasse. Time series prediction using dirrec strategy. volume 6, pages 143–148, 01
2006.

[60] Gianluca Bontempi. Long term time series prediction with multi-input multi-output local learning. _Proceedings of_
_the 2nd European Symposium on Time Series Prediction (TSP), ESTSP08_, 01 2008.

[61] Souhaib Ben Taieb, Gianluca Bontempi, Antti Sorjamaa, and Amaury Lendasse. Long-term prediction of time
series by combining direct and mimo strategies. _2009 International Joint Conference on Neural Networks_, pages
3054–3061, 2009.

[62] F. M. Bianchi, E. De Santis, A. Rizzi, and A. Sadeghian. Short-term electric load forecasting using echo state
networks and pca decomposition. _IEEE Access_, 3:1931–1943, 2015.

[63] Ilya Sutskever, Oriol Vinyals, and Quoc V. Le. Sequence to sequence learning with neural networks. In _Advances_
_in Neural Information Processing Systems 27: Annual Conference on Neural Information Processing Systems_
_2014, December 8-13 2014, Montreal, Quebec, Canada_, pages 3104–3112, 2014.

[64] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align
and translate. _CoRR_, abs/1409.0473, 2014.

[65] Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V. Le, Mohammad Norouzi, Wolfgang Macherey, Maxim
Krikun, Yuan Cao, Qin Gao, Klaus Macherey, Jeff Klingner, Apurva Shah, Melvin Johnson, Xiaobing Liu, ukasz
Kaiser, Stephan Gouws, Yoshikiyo Kato, Taku Kudo, Hideto Kazawa, and Jeffrey Dean. Google’s neural machine
translation system: Bridging the gap between human and machine translation. 09 2016.

[66] A. Graves, A. Mohamed, and G. Hinton. Speech recognition with deep recurrent neural networks. In _2013 IEEE_
_International Conference on Acoustics, Speech and Signal Processing_, pages 6645–6649, May 2013.

[67] Jan K Chorowski, Dzmitry Bahdanau, Dmitriy Serdyuk, Kyunghyun Cho, and Yoshua Bengio. Attention-based
models for speech recognition. In C. Cortes, N. D. Lawrence, D. D. Lee, M. Sugiyama, and R. Garnett, editors,
_Advances in Neural Information Processing Systems 28_, pages 577–585. Curran Associates, Inc., 2015.

[68] D. Bahdanau, J. Chorowski, D. Serdyuk, P. Brakel, and Y. Bengio. End-to-end attention-based large vocabulary
speech recognition. In _2016 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_,
pages 4945–4949, March 2016.

[69] Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhudinov, Rich Zemel, and
Yoshua Bengio. Show, attend and tell: Neural image caption generation with visual attention. In Francis Bach
and David Blei, editors, _Proceedings of the 32nd International Conference on Machine Learning_, volume 37 of
_Proceedings of Machine Learning Research_, pages 2048–2057, Lille, France, 07–09 Jul 2015. PMLR.

[70] Ronald J. Williams and David Zipser. A learning algorithm for continually running fully recurrent neural networks.
_Neural Computation_, 1:270–280, 1989.


24


A PREPRINT


[71] Marc’Aurelio Ranzato, Sumit Chopra, Michael Auli, and Wojciech Zaremba. Sequence level training with
recurrent neural networks. _CoRR_, abs/1511.06732, 2015.

[72] Samy Bengio, Oriol Vinyals, Navdeep Jaitly, and Noam Shazeer. Scheduled sampling for sequence prediction
with recurrent neural networks. In _Proceedings of the 28th International Conference on Neural Information_
_Processing Systems - Volume 1_, NIPS’15, pages 1171–1179, Cambridge, MA, USA, 2015. MIT Press.

[73] Alex Lamb, Anirudh Goyal, Ying Zhang, Saizheng Zhang, Aaron C. Courville, and Yoshua Bengio. Professor
forcing: A new algorithm for training recurrent networks. In _NIPS_, 2016.

[74] Yaser Keneshloo, Tian Shi, Naren Ramakrishnan, and Chandan K. Reddy. Deep reinforcement learning for
sequence to sequence models. _CoRR_, abs/1805.09461, 2018.

[75] Henning Wilms, Marco Cupelli, and A Monti. Combining auto-regression with exogenous variables in sequenceto-sequence recurrent neural networks for short-term load forecasting. pages 673–679, 07 2018.

[76] Yann Lecun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document
recognition. In _Proceedings of the IEEE_, pages 2278–2324, 1998.

[77] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural
networks. In F. Pereira, C. J. C. Burges, L. Bottou, and K. Q. Weinberger, editors, _Advances in Neural Information_
_Processing Systems 25_, pages 1097–1105. Curran Associates, Inc., 2012.

[78] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition.
_CoRR_, abs/1409.1556, 2014.

[79] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan,
Vincent Vanhoucke, and Andrew Rabinovich. Going deeper with convolutions. In _The IEEE Conference on_
_Computer Vision and Pattern Recognition (CVPR)_, June 2015.

[80] Ross Girshick. Fast r-cnn. In _Proceedings of the IEEE international conference on computer vision_, pages
1440–1448, 2015.

[81] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards real-time object detection with
region proposal networks. In _Advances in neural information processing systems_, pages 91–99, 2015.

[82] Max Jaderberg, Karen Simonyan, Andrew Zisserman, et al. Spatial transformer networks. In _Advances in neural_
_information processing systems_, pages 2017–2025, 2015.

[83] Ryan Dahl, Mohammad Norouzi, and Jonathon Shlens. Pixel recursive super resolution. _arXiv preprint_
_arXiv:1702.00783_, 2017.

[84] Christian Ledig, Lucas Theis, Ferenc Huszár, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew
Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, et al. Photo-realistic single image super-resolution using a
generative adversarial network. In _2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_,
pages 105–114. IEEE, 2017.

[85] Aäron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alexander Graves, Nal
Kalchbrenner, Andrew Senior, and Koray Kavukcuoglu. Wavenet: A generative model for raw audio. In _Arxiv_,
2016.

[86] Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, and Yann N Dauphin. Convolutional sequence to
sequence learning. _arXiv preprint arXiv:1705.03122_, 2017.

[87] Anastasia Borovykh, Sander Bohte, and Kees Oosterlee. Conditional time series forecasting with convolutional
neural networks. In _Lecture Notes in Computer Science/Lecture Notes in Artificial Intelligence_, pages 729–730,
September 2017.

[88] Vincent Dumoulin and Francesco Visin. A guide to convolution arithmetic for deep learning. _CoRR_,
abs/1603.07285, 2016.

[89] Shaojie Bai, J Zico Kolter, and Vladlen Koltun. An empirical evaluation of generic convolutional and recurrent
networks for sequence modeling. _arXiv preprint arXiv:1803.01271_, 2018.

[90] François Chollet et al. Keras. `[https://github.com/fchollet/keras](https://github.com/fchollet/keras)`, 2015.

[91] Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy
Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow, Andrew Harp, Geoffrey Irving, Michael
Isard, Yangqing Jia, Rafal Jozefowicz, Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dandelion Mané, Rajat
Monga, Sherry Moore, Derek Murray, Chris Olah, Mike Schuster, Jonathon Shlens, Benoit Steiner, Ilya Sutskever,
Kunal Talwar, Paul Tucker, Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viégas, Oriol Vinyals, Pete Warden,
Martin Wattenberg, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng. TensorFlow: Large-scale machine learning on
heterogeneous systems, 2015. Software available from tensorflow.org.


25


A PREPRINT


[92] Christopher M. Bishop. _Pattern Recognition and Machine Learning (Information Science and Statistics)_ . SpringerVerlag, Berlin, Heidelberg, 2006.

[93] Sihan Li, Jiantao Jiao, Yanjun Han, and Tsachy Weissman. Demystifying resnet. _arXiv preprint arXiv:1611.01186_,
2016.

[94] Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal
covariate shift. pages 448–456, 2015.

[95] A. Marinescu, C. Harris, I. Dusparic, S. Clarke, and V. Cahill. Residential electrical demand forecasting in very
small scale: An evaluation of forecasting methods. In _2013 2nd International Workshop on Software Engineering_
_Challenges for the Smart Grid (SE4SG)_, pages 25–32, May 2013.


26


