## **hls4ml: An Open-Source Codesign Workflow to Empower** **Scientific Low-Power Machine Learning Devices**



Farah Fahim [∗]

Benjamin Hawks
Christian Herwig
James Hirschauer
Sergo Jindariani
Nhan Tran [∗]


Fermilab

Batavia, IL, USA


Manuel Blanco Valentin

Josiah Hester
Yingyi Luo
John Mamish
Seda Orgrenci-Memik
Northwestern University
Evanston, IL, USA


Scott Hauck

Shih-Chieh Hsu

University of Washington
Seattle, WA, USA


Duc Hoang
Rhodes College
Memphis, TN, USA


**ABSTRACT**



Luca P. Carloni
Giuseppe Di Guglielmo
Columbia University
New York, NY, USA


Thea Aarrestad

Hamza Javed
Vladimir Loncar

Maurizio Pierini

Adrian Alan Pol

Sioni Summers

European Organization for Nuclear
Research (CERN)
Geneva, Switzerland


Jennifer Ngadiuba
Caltech

Pasadena, CA, USA


Edward Kreinar

HawkEye360
Herndon, VA, USA



Philip Harris
Jeffrey Krupa
Dylan Rankin
MIT

Cambridge, MA, USA


Javier Duarte
UC San Diego
La Jolla, CA, USA
jduarte@ucsd.edu


Mia Liu

Purdue University
West Lafayette, IN, USA


Zhenbin Wu

University of Illinois at Chicago
Chicago, IL, USA



Accessible machine learning algorithms, software, and diagnostic tools for energy-efficient devices and systems are extremely
valuable across a broad range of application domains. In scientific
domains, real-time near-sensor processing can drastically improve
experimental design and accelerate scientific discoveries. To support domain scientists, we have developed hls4ml, an open-source
software-hardware codesign workflow to interpret and translate
machine learning algorithms for implementation with both FPGA
and ASIC technologies. In this paper, we describe the essential
features of the hls4ml workflow including network optimization


∗ Also affiliated with Northwestern University


Permission to make digital or hard copies of part or all of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for third-party components of this work must be honored.
For all other uses, contact the owner/author(s).
_TinyML Research Symposium’21, March 2021, San Jose, CA_
© 2021 Copyright held by the owner/author(s).



techniques—such as pruning and quantization-aware training—
which can be incorporated naturally into the device implementations. We expand on previous hls4ml work by extending capabilities and techniques towards low-power implementations and
increased usability: new Python APIs, quantization-aware pruning, end-to-end FPGA workflows, long pipeline kernels for low
power, and new device backends include an ASIC workflow. Taken
together, these and continued efforts in hls4ml will arm a new generation of domain scientists with accessible, efficient, and powerful
tools for machine-learning-accelerated discovery.


**KEYWORDS**


hls4ml, machine learning, neural networks, tinyML, FPGA, ASIC,
low-power, low-latency


**ACM Reference Format:**

Farah Fahim, Benjamin Hawks, Christian Herwig, James Hirschauer, Sergo
Jindariani, Nhan Tran, Luca P. Carloni, Giuseppe Di Guglielmo, Philip Harris,
Jeffrey Krupa, Dylan Rankin, Manuel Blanco Valentin, Josiah Hester, Yingyi
Luo, John Mamish, Seda Orgrenci-Memik, Thea Aarrestad, Hamza Javed,
Vladimir Loncar, Maurizio Pierini, Adrian Alan Pol, Sioni Summers, Javier
Duarte, Scott Hauck, Shih-Chieh Hsu, Jennifer Ngadiuba, Mia Liu, Duc


TinyML Research Symposium’21, March 2021, San Jose, CA Fahim, et al.



Hoang, Edward Kreinar, and Zhenbin Wu. 2021. hls4ml: An Open-Source
Codesign Workflow to Empower Scientific Low-Power Machine Learning
Devices. In _Proceedings of TinyML Research Symposium (TinyML Research_
_Symposium’21)._ ACM, New York, NY, USA, 10 pages.


**1** **INTRODUCTION**


The efficient implementations of machine learning (ML) algorithms
in dedicated hardware devices at the “edge,” or near-sensor, offers
numerous advantages. Edge processing and data compression can
greatly reduce data rates and the energy required for data movement. Furthermore, real-time data processing and interpretation
can greatly accelerate decision-making, hypothesis testing and even
enable just-in-time interventions.
Staggering data rates and massive datasets are generated across
a broad range of modern scientific applications in high energy
physics, material science, and astrophysics. For example at the
CERN Large Hadron Collider (LHC), experiments typically produce
data at rates of Pb/s, and at the Fermilab accelerator complex, hundreds of thousands of devices monitor miles of beamlines that steer

near speed-of-light particle beams. In these physics experiments,
low-latency ML is required for real-time decision making with a
range with a range of requirements from tens of nanoseconds to submillisecond. In many ways, techniques for resource-constrained
ML implementations are similar whether targeting low power or
ultra low latency and high throughput. **In this paper, we discuss**
**how tools developed for low-latency applications in science**
**could be deployed for low-power applications.**


_**Demand for accessible tools**_ _._ Low-power ML is in demand in
scientific, industrial, and commercial computing [ 1 ]. Fitness bands,
smartwatches, and other wearables that capture human health
and behaviors from complex and multivariate continuous sensor
data [ 2 ], wireless sensor networks deployed for tracking and understanding threatened animal populations using imaging and other
sensors [ 3 ], and even large-scale wireless agriculture sensing [ 4 ]
all necessitate powerful local computation on a budget. Despite
the broad need for local, low-power ML and the growing number of edge devices and scientific applications, general-purpose
off-the-shelf hardware has not kept pace with these computing
demands. The challenge for domain scientists is that a broad range
of expertise is required to arrive at full ML device implementations.
This requires significant resources and a team of domain scientists,
computer scientists, and engineers. Typically, domain scientists
may be experts in domain ML, their experimental devices, or system engineering techniques, but very rarely in all requisite areas
simultaneously.
To tackle this challenge, we need a framework which makes digital hardware implementations of ML algorithms more _accessible,_
_interpretable, and (re)usable_ to domain scientists. While ultimately
hardware implementations require completely engineered solutions, allowing domain scientists to codesign algorithms based on
their system and application constraints is extremely valuable in
reducing engineering time and enabling faster design iterations.
Optimizing this process will greatly reduce the _time to science_ . Finally, to cater to both large experimental collaborations and smaller
laboratory groups, the tools should be as open-source as possible.



_**ML-hardware codesign tools**_ _._ Software like TensorFlow and
PyTorch have democratized ML for scientists, lowering the timeto-science across domains. In developing hls4ml as an open-source
codesign workflow [ 5, 6 ], our main goal has been to augment popular ML frameworks with an effective path to efficient hardware
implementations. After a user trains their ML algorithms in common ML software frameworks, hls4ml translates them into digital implementations using high-level synthesis (HLS) tools [ 7 ] for
energy-efficient devices, which can be realized with technologies
such as field-programmable gate arrays (FPGAs) and applicationspecific integrated circuits (ASICs). The choice of the HLS paradigm
is motivated by the shorter design cycle and similar quality of result compared to register-transfer level (RTL) flows [ 8 ]. With the
introduction of an open-source framework like hls4ml, “tinyML”
techniques can be made accessible to nonexperts. The benefit of
hls4ml is two-fold: it lets nonexperts create bespoke, cutting-edge
ML accelerators for low-power and low-latency systems, and it lets
nonexperts develop intuition about how their design choices affect
system power consumption.
Several recent results highlight the power of the hls4ml approach including support for quantization down to binary and
ternary precision [ 9 ], pruning, tunable parallelization [ 5 ], boosted
decision trees [ 10 ], quantization-aware training (QAT) [ 11 ], and
convolutional [12] and graph neural networks (NNs) [13]. The development of hls4ml is application driven. While originally its
main target was low-latency applications, recent work has focused
on opportunities for longer latency, low-power applications. Algorithm design with hls4ml involves the generation of custom
firmware for a specified NN architecture. The customized design
ensures an efficient use of resources that is essential to run in lowlatency, resource-constrained systems, and is useful across a broader
range of applications. **This paper reviews salient core features**
**of hls4ml and extends previous work [5, 9–12] by present-**
**ing a number of recently added features that aid in targeting**
**low-power systems** : initial results of quantization-aware pruning
(QAP) [ 14 ], full end-to-end workflows that embed hls4ml algorithms into Vitis Accel designs for Xilinx FPGAs, new matrix-vector
kernels which are optimized for longer pipeline intervals, sparse
operations, and low power, and support for multiple HLS compilers
and devices, including ASICs. Our approach for low-power devices
focuses on ML for FPGAs and ASICs as energy efficient hardware
architectures [ 15, 16 ]. The hls4ml framework aims to bridge novel
techniques in NN inference optimization and hardware implementation while making it accessible to domain scientists.
This paper is organized as follows. In Sec. 2, we discuss the
hls4ml workflow and features for introspection, validation, and
support for multiple device types. In Sec. 3, we discuss codesign
techniques developed at ML training to develop optimal hardware
implementations. In Sec. 4, we describe how those NNs get implemented in hardware and the available configurations to the user.
Finally, we summarize and present an outlook in Sec. 5.


hls4ml: An Open-Source Codesign Workflow to Empower Scientific Low-Power Machine Learning Devices TinyML Research Symposium’21, March 2021, San Jose, CA



**Related Work**


Other open-source efforts have explored ML on edge devices, including FPGAs, mobile devices, and microcontrollers with integrated workflows from training to deployment. Surveys of existing toolflows can be found in Refs. [ 17 – 20 ]. The fpgaConvNet
library [ 21 – 24 ] converts CNNs specified in Caffe [ 25 ] or Torch
formats into high-level specifications of streaming accelerators
that can be synthesized to hardware with Xilinx Vivado HLS. FPDNN [ 26 ] is a framework that takes TensorFlow [ 27 ]-described
CNNs as input, and generates the hardware implementations on
FPGA boards with RTL-HLS hybrid templates. DNNWeaver [ 28 ]
is an open-source project that supports CNNs specified in Caffe
and generates Verilog code using hand-optimized Verilog templates
with a high degree of portability. Caffeine [ 29 ] is a CNN accelerator
for Caffe models targeting Xilinx coprocessors over a PCIe interface. Snowflake [ 30 ] is a CNN accelerator with models specified
in Torch [ 31 ] and a single, sequential computation architecture
designed to perform at near-peak hardware utilization targeting
Xilinx system-on-chips (SoCs). K. Majumder et al. (2019) propose
an FPGA-based accelerator design to execute CNNs that leverages
TensorFlow for model description and exploits reuse along all
dimensions with a 1D systolic array of processing elements [ 32 ]. A.
Rahman et al. (2016) present a flexible 3D neuron array architecture
for CNNs on FPGAs with optimization techniques for a given set
of FPGA resource constraints [ 33 ]. Vitis AI [ 34 ] is Xilinx’s development platform for AI inference on Xilinx devices, consisting of
optimized intellectual property (IP) cores, tools, libraries, models,
and example designs. The FINN project [ 35 – 37 ] is a framework from
Xilinx Research Labs to explore quantized deep NN inference on
FPGAs, with emphasis on generating dataflow-style architectures
customized for each network. It includes tools for training quantized NNs (QNNs) such as Brevitas [ 38 ], the FINN compiler, and
the finn-hlslib Vivado HLS library of FPGA components for QNNs.
Further, TensorFlow Lite [ 39 ] is a set of tools to help developers
run TensorFlow [ 27 ] models on mobile, embedded, and internet
of things (IoT) devices. It currently supports Android, iOS, and
Linux devices (like Raspberry Pi), as well as microcontrollers (like
Arduinos). It enables on-device ML inference with low latency and
a small binary size. FixyNN [ 40 ], and specifically the open-source
DeepFreeze [ 41 ], allow for the automatic generation and optimization of a fixed-weight feature extractor plus a programmable CNN
hardware accelerator from a TensorFlow description for mobile
devices. Hacene et al. (2020) propose a combination of a pruning
technique and a quantization scheme that reduces the complexity
and memory usage of convolutional layers, by replacing the convolutional operation by a low-cost multiplexer [ 42 ]. In particular, the
authors propose an efficient hardware architecture implemented on
FPGA on-chip memory. Chang et al. (2021) apply different quantization schemes (fixed-point and sum-power-of-two) to different rows
of the weight matrix to achieve better utilization of heterogeneous
FPGA hardware resources [43].


**2** **HLS4ML WORKFLOW**


The hls4ml workflow performs automatically the task of translating a trained NN, specified by the model’s architecture, weights,
and biases, into the specification of a hardware accelerator that can



**boxes (left) describe the model training and compression**
**steps performed within conventional ML software frame-**
**works. The hls4ml configuration and conversion steps are**
**shown in the blue boxes (center). The black boxes (right)**
**illustrate possible ways to export and integrate the HLS**
**project into a larger hardware design.**


be synthesized with HLS tools. Figure 1 shows the schematic of a
typical workflow. The first part of the workflow illustrated in red
depicts the usual steps required to design a NN for a specific task.
This component, performed with tools like (Q)Keras and PyTorch,
involves a training step and possible compression steps (explained
in detail in Sec. 3) before converging on a final model. The blue
section of the workflow is performed with hls4ml, which translates
a model into an HLS project that can subsequently be synthesized
and implemented on an FPGA or ASIC, as depicted by the black

section.

At a high level, FPGA and ASIC algorithm design is different
from programming a CPU in that independent operations may
run fully in parallel or concurrently. Furthermore, independent
operations may be pipelined such that the algorithm can accept
new inputs while it is still operating on previous inputs. However,
such operations consume dedicated resources onboard the FPGA
or ASIC and cannot be dynamically remapped while running. The
challenge in creating an optimal digital design is to balance available
resources with achieving the power, latency, throughput goals of
the target application.
The hls4ml workflow provides a number of configurable parameters which can help the user explore and customize the space
of latency, throughput, power, and resource usage tradeoffs for
their application. Because every application is different, the goal
of the hls4ml package is to empower the user to perform this optimization through automated NN translation and design iteration.
hls4ml leverages HLS to generate hardware modules from code
written in high-level programming languages like C/C++ [ 44 ]. Each
layer and activation type is implemented as a separate configurable
module customized to perform that specific operation. During the
hls4ml conversion, these modules are composed in the correct way
to perform the inference for a full ML model. Computation for each
layer of the network is carried out in separate hardware, and the
full model is synthesized into an IP core that can be integrated
into a full application. Large throughput and low latency can be


















TinyML Research Symposium’21, March 2021, San Jose, CA Fahim, et al.



**Figure 2: Internal structure of the hls4ml package. Model**
**converters translate models from Keras, PyTorch, etc. into**
**an intermediate HLSModel representation. This represen-**
**tation can be further configured and optimized. Different**
**backend writers can be used to export the model into a given**
**vendor-specific language, such as Vitis HLS, Quartus HLS,**
**Catapult HLS, or others.**


achieved by pipelining data through the network. Furthermore,
resource usage can be optimized because each layer is tailored during conversion to the specific model and, if desired, set of weights.
This optimization extends to zero suppression, where the layer can
be configured to skip multiplications by zero weights. Although
it may lead to slightly less optimal performance than RTL-based
design, HLS-based design has significant benefits: it raises the level
of abstraction, reduces the iteration time, simplifies the validation
phase, and enables greater exploration and evaluation of design
alternatives.


**Package Architecture**


To provide flexibility and ease-of-use, we implemented hls4ml as
a Python package that provides both a programming API and
visualization capabilities. Figure 2 shows the internal structure
of the hls4ml Python package. The package first converts the
user-specified model into a common internal representation of the
network graph. Converters are provided for (Q)Keras, TensorFlow, PyTorch, and ONNX model formats. At the conversion step,
the user-provided configuration is also attached to the model. For
a NN trained with QKeras quantization-aware training (QAT), the
quantization settings of the model are propagated into the hls4ml
internal representation.
A suite of _optimizers_ then modify the network graph to target a
more lightweight, faster inference. At this step, for example, batch
normalization [ 45 ] layers are fused with the preceding dense or
convolutional layer, or with the subsequent binary (or ternary)
tanh activation layer. Where possible, these optimizations precompute quantities needed during inference involving constant model
parameters, in order to reduce operations at runtime.
The hls4ml model object can be inspected, and the package
provides a set of utilities to aid the configuration process. These
include a visualization tool to display the NN graph decorated with
the applied user configuration, and tools to numerically profile the
model which can help guide the user settings, e.g. for bit precision.



Figure 3 shows an example of the numerical profiling output
from hls4ml for a fully-connected NN of a benchmark autoencoder
architecture for anomaly detection [ 46 ]. Each hidden layer is composed of a dense layer, batch normalization, and a rectified linear
unit (ReLU) [ 47, 48 ] activation function. The distribution of the
weight values is represented by a boxplot, showing the range covering the bulk of the distribution as well as the extremities. On top
of this, the user-provided precision configuration is shown with
the grey boxes. Generally, it is crucial that the largest absolute valued weights can be represented with the chosen type (the boxes
overlap at the right of the plot). There is some flexibility to reduce
the precision by truncating small valued weights, with minimal
impact on accuracy. This additional visualization tool can be used
to quickly tune the configuration for more efficient inference.
One key feature of the programming API is the capability to execute the bit-accurate emulation of the generated HLS-synthesizable
code in the Python environment, for example as a Jupyter Notebook. In conventional HLS-design flows, developers craft C/C++
testbenches which they execute in the HLS-vendor simulation environment to verify algorithm performance. The hls4ml API enables
a workflow that will be much more familiar to ML developers, where
inference can be performed on tensor or array data in Python code,
providing the opportunity to complete a detailed analysis. In addition to evaluating the hls4ml model output, users can access
the detailed output of any hidden layer of the network, which can
aid in debugging and performing hyperparameter optimization for
quantized models. When the hls4ml model is written out, the backend maps the graph onto its library of optimized inference code.
This inference code can run on the CPU executing the conversion,
in order to check numerical correctness against the original NN.
After this step, the user runs the vendor synthesis tools in order to
produce an IP core, and evaluate latency, throughput, and resources.
Presently, the most advanced backend is for Xilinx Vivado HLS,
with codebases optimized for Intel Quartus HLS [ 49 ] and Mentor
Catapult HLS [50] under active development.


**3** **NEURAL NETWORK TRAINING AND**

**OPTIMIZATION**


Reducing the precision of the calculations in the NN and removing
unimportant calculations can drastically improve the efficiency of
the NN implementation with little to no loss in performance. While
applying these changes to a model post-training can be successful,
to be maximally effective, we should consider these effects at the
time of NN training.
We consider two benchmark tasks to demonstrate the versatil
ity of model optimization in the hls4ml workflow. The first is a
high-energy particle jet classification task on a dataset [ 5, 51, 52 ]
consisting of 16 features for simulated particle jets produced in
proton-proton collisions and originating from one of five classes of
particles: W or Z bosons, light quarks, top quarks, or gluons. The
baseline model is a simple fully-connected NN with three hidden
layers, with 64, 32, and 32 nodes, respectively, activated by ReLUs.
The second benchmark is the MNIST handwritten digit classification task [ 53 ]. The baseline model we consider is a fully-connected
NN with one hidden layer with 16 nodes and ReLU activation.


hls4ml: An Open-Source Codesign Workflow to Empower Scientific Low-Power Machine Learning Devices TinyML Research Symposium’21, March 2021, San Jose, CA

















**3.1** **Quantization-Aware Training**


Quantized [ 54 – 59 ] and even binarized [ 9, 57, 60 – 63 ] NNs have been
studied as a way to compress NNs by reducing the number of bits
required to represent each weight. FPGAs provide considerable freedom in the choice of data type and precision. Both choices should
be considered carefully to prevent squandering FPGA resources
and incurring additional latency. In hls4ml, we use fixed-point
arithmetic, which requires less resources and has a lower latency
than floating-point arithmetic. The inputs, weights, biases, sums,
and outputs of each layer are all represented as fixed-point numbers. For each, the number of bits used to represent the integer
and fractional part can be configured separately for the use case.
The precision can be reduced significantly without causing a loss
in performance [ 57 ]. We determine the number of bits to assign
for the fractional part by scanning the performance values as a
function of the bit precision.
One simple way to reduce a model’s size is through post-training
quantization (PTQ) where pre-trained model parameters are clipped
or rounded to lower precision. However, this process is lossy and
sacrifices model performance. To solve this, QAT has been proposed [ 60, 64, 65 ]. In these approaches, the reduced precision of
the weights and biases are accounted for directly in the training of
the NN. In QKeras, this is implemented using the straight-through
estimator (STE) [ 60 ], where the forward pass of the training applies
the quantization, while the backward pass assumes that quantization is the identity function, as the quantization function is not
differentiable. It has been found that QAT is even more efficient
than PTQ while retaining the same performance. In these studies, the same type of quantization is applied everywhere. More



**Figure 4: Performance of quantization-aware training from**
**Ref. [11] in terms of the relative accuracy as a function of**
**bit width. The relative accuracy is evaluated with respect**
**to the floating-point baseline model. The CPU-based emu-**
**lation (solid green) of the FPGA-based QAT model (solid or-**
**ange) is compared to the PTQ model (dashed purple).**


recently [ 66, 67 ], it has been noted that some layers may accommodate extreme quantization better than other layers, suggesting
that per-layer heterogeneous quantization is the optimal way to
achieve high accuracy at low resource cost.
An example of the power of QAT is shown in Fig. 4 from Ref. [ 11 ]
which uses QKeras. For the particle physics task with a fullyconnected NN, the accuracy of the reduced precision model is
compared to the 32-bit floating-point implementation as the bit
width is scanned. In the PTQ case, the accuracy begins to drop
below 14 bits, while in the QAT case the accuracy is comparable to
the 32-bit floating implementation down to 6 bits. More detailed
discussion on layer-by-layer quantization is presented in Ref. [ 11 ].
In Section 4, we discuss the implementation of QAT in hls4ml and
its effect in terms of on-chip resources.


**3.2** **Quantization-Aware Pruning**


Network compression is a common technique to reduce the size,
energy consumption, and overtraining of deep NNs [ 58 ]. Several approaches have been successfully deployed to compress networks [ 68 –
70 ]. Here we focus specifically on _parameter pruning_ : the selective
removal of weights based on a particular ranking [58, 71–75].
Prior studies have combined pruning and quantization trivially
by pruning 32-bit floating-point models and applying post-training
quantization. One such approach, whose results are shown in Sec. 4.2,
consists of iterative parameter pruning and retraining of a 32-bit
floating-point model [ 5, 58, 76 ] with _𝐿_ 1 regularization, where the
loss function is augmented with an additional penalty term _𝜆_ ∥ _𝒘_ ∥ 1,
where _𝒘_ is a vector of all of the model weights and _𝜆_ is a tunable hyperparameter. _𝐿_ 1 regularization produces sparse models, provides
built-in feature selection [ 77 ], and is readily available in many ML
workflows. After training the model with _𝐿_ 1 regularization with a


TinyML Research Symposium’21, March 2021, San Jose, CA Fahim, et al.




|Model bit width 16 (PTQ) 14 (PTQ) 6 (QAT)|Col2|
|---|---|
|Accuracy [%]<br>Latency [ns]<br>DSP [%]<br>LUT [%]<br>FF [%]|76.4<br>75.8<br>76.2<br>50<br>50<br>45<br>27 (1,852)<br>17 (1,132)<br>0.6 (38)<br>7.5 (89,209)<br>7.6 (90,019)<br>5.4 (63,251)<br>0.5 (11,266)<br>0.4 (9,262)<br>0.2 (4,394)|












|1.0|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
|.8<br>.9<br>.0|||||
|.8<br>.9<br>.0||**Percent **|**pruned (approx.)**<br>**98.8%**<br>|**pruned (approx.)**<br>**98.8%**<br>|
|.8<br>.9<br>.0|||~~**96.6%**~~<br>**90.0%**<br>**80.0%**<br>|~~**96.6%**~~<br>**90.0%**<br>**80.0%**<br>|
|.5<br>.6<br>.7|||~~**70.0%**~~<br>**60.0%**<br>**50.0%**<br>|~~**70.0%**~~<br>**60.0%**<br>**50.0%**<br>|
|.5<br>.6<br>.7|Lo|ttery ticket pruning<br>32-bit<br>|~~**40.0%**~~<br>**30.0%**<br>**20.0%**<br>**10.0%**<br>|~~**40.0%**~~<br>**30.0%**<br>**20.0%**<br>**10.0%**<br>|
|10<br>~~5~~<br>||6-bit|**0.0%**|**0.0%**|
|10<br>~~5~~<br>||10<br>~~6~~<br>10<br>7<br>|10<br>~~6~~<br>10<br>7<br>|10<br>~~6~~<br>10<br>7<br>|



**Figure 5: Performance of quantization-aware pruning using**
**the lottery ticket pruning scheme as a function of hardware**
**computational complexity. After QAP, the 6-bit, 80% pruned**
**model achieves a factor of 50 reduction in BOPs compared**
**to the 32-bit, unpruned model with no loss in performance.**


small _𝜆_ (e.g. 10 [−][4] ), the weights are sorted based on their absolute
value relative to the maximum absolute value of the weights in that
particular layer. Weights falling below a certain percentile are removed. The model can then be trained again with _𝐿_ 1 regularization
while masking the previously pruned weights. This process can
be iterated several times until the desired level of compression is
reached.

While the above approach is effective, we describe here an alternative approach based on the lottery ticket (LT) hypothesis [ 73 ]
where the remaining weights after each pruning step are initialized
back to their original values (“weight rewinding”). We refer to this
method as LT pruning. We also propose a new hybrid method [ 14 ]
for constructing efficient NNs, quantization-aware pruning (QAP),
which combines a pruning procedure with training that accounts for
quantized weights. As a first demonstration, we use Brevitas [ 38 ]
to perform QAT and iteratively prune a fraction of the weights
following the LT method of weight rewinding.
This is done for the jet classification task presented in the previous section. At each training iteration, roughly 10% of the original
network is pruned. The results of pruning with this method are
shown in Fig. 5 for a 6-bit fixed-point version of the network compared to the 32-bit floating-point model. The performance in terms
of the area under the curve (AUC) is shown as a function of bit
operations (BOPs) [78], defined per-layer as


BOPs = _𝑚𝑛_ ((1 − _𝑓_ _𝑝_ ) _𝑏_ _𝒂_ _𝑏_ _𝒘_ + _𝑏_ _𝒂_ + _𝑏_ _𝒘_ + log 2 ( _𝑛_ )) (1)


where _𝑛_ ( _𝑚_ ) is the number of inputs (outputs), _𝑏_ _𝒘_ ( _𝑏_ _𝒂_ ) is the bit
width of the weights (activations), and _𝑓_ _𝑝_ is the fraction of pruned
layer weights. BOPs are a measure of the hardware computational
complexity of a quantized NN after pruning. In Sec. 3.1 we found
that a 6-bit implementation of this network sacrificed no performance. Here, we find that pruning the 6-bit network by 80% using
QAP still maintains the same performance as the 32-bit version.



**Table 1: Model accuracy, latency and resource utilization for**
**16-, 14-, and 6-bit models. Resources are listed as a percent-**
**age of available resources, with absolute numbers quoted**
**in parenthesis, for a Xilinx Virtex UltraScale+ VU9P FPGA**
**with a clock frequency of 200 MHz using hls4ml v0.5.0 and**
**Vivado HLS 2019.2**


**4** **DIGITAL IMPLEMENTATION ELEMENTS**


Following the training-time optimizations described in the previous section, we describe important elements for deploying those
optimized NNs in efficient digital implementations.


**4.1** **Quantization with a QKeras Frontend**


Reducing precision saves resources used for signal routing as well
as resources and latency used for mathematical operations. For
example, the limiting resource for many FPGA applications is the
number of DSPs, which are used primarily for multiplications. The
number of DSPs used per multiplier depends on the precision of the
numbers being multiplied and can change abruptly. For example,
one Xilinx DSP48E1 block [ 79 ] can multiply a 25-bit number with
an 18-bit number, but two are required to multiply a 25-bit number
with a 19-bit number. Similarly, the latency of multipliers increases
with precision, though they can remain pipelined.
To allow for automated translation of a QKeras model to RTL,
hls4ml has been extended to interpret and optimize quantized
QKeras layer types. When converting a QKeras model into an HLS
project, the model quantization configuration is passed to hls4ml
and enforced in the FPGA firmware. This ensures that the use of
specific, arbitrary precision in the QKeras model is maintained
during inference. For example, when using a quantizer with a given
rescaling parameter _𝛼_, hls4ml inserts an operation to rescale the
layer output. For binary and ternary weights and activations, the
same strategies as in Ref. [ 9 ] are used. With binary layers, the
arithmetical value of “-1” is encoded as “0,” allowing the product to
be expressed as an XNOR operation.
As an example of the integration of QKeras and hls4ml, we
now describe an FPGA implementation of the model presented in
Sec. 3.1. The FPGA implementation results are reported in Table 1
for the 16- and 14-bit PTQ and 6-bit QAT models. The effect of QAT
is that the FPGA resources are drastically reduced, especially in
the case of DSPs. In Ref. [ 11 ], a more detailed exploration of model
implementations is presented, including per-layer optimizations.
The QKeras+ hls4ml framework extends to large CNNs. Specifically, Ref. [ 12 ] demonstrates support for large CNN architectures
through a stream-based implementation of convolutional and pooling layers using first in, first out (FIFO) buffers. A benchmark CNN
classifier trained on the Street View House Numbers Dataset is compressed through pruning and QAT to reduce the FPGA resource
utilization while retaining the floating-point model accuracy. Once


hls4ml: An Open-Source Codesign Workflow to Empower Scientific Low-Power Machine Learning Devices TinyML Research Symposium’21, March 2021, San Jose, CA



implemented on the FPGA, the model achieves a 5 _𝜇_ s latency, while
consuming less than 10% of the FPGA resources and low estimated
power [16].


**4.2** **Parallelization and Sparsity**


The core component of dense and convolutional NN layer implementations in hls4ml is a matrix-vector multiplication kernel. In
addition to the precision at which these kernels are executed there
are further configurations that can be used to tune the digital design
for a specific task. We explore two of them here: parallelization and
sparsity.


_Parallelization._ A matrix-vector multiplication kernel requires a
number of multiplication operations based on the dimensions of
the matrix. The trade off between latency, throughput and FPGA
resource usage is determined by the parallelization of the inference
calculation and the number of multiplications performed in parallel.
In hls4ml, this is configured with a “reuse factor” that sets the
number of times a multiplier is used in the computation of a layer’s
output values. With a reuse factor of one, the computation is fully
parallel, i.e. each multiplier is used once. With a reuse factor of
_𝑅_, 1 / _𝑅_ of the computation is done at a time with a factor of 1 / _𝑅_
fewer multipliers. To make routing more convenient, often there
are preferred values of _𝑅_ depending on the dimensions of the matrix
itself.

The matrix-vector multiplication kernel cannot accept new inputs until all of the previous multiplications have been performed,
a period of time known as the initiation interval (II). For larger
reuse factors, the matrix-vector multiplication kernel has longer
latency and II, but uses less on-chip resources. In hls4ml, we implement each layer calculation independently and sequentially. The
calculation of one layer cannot be initiated until the calculation of
the previous layer has completed. Therefore, the total latency is
equal to the sum of latencies of each layer plus the latency required
to connect the layers. The processing throughput, i.e. the number
of inferences per unit time, is inversely proportional to the reuse
factor.

The configurability of the reuse factor allows users of hls4ml
to tune their hardware implementation for their system requirements. In Fig. 6, we show the FPGA resources for a dense, fullyconnected NN which is used for the MNIST handwritten digit classification task [ 53 ]. The total number of multiplications needed
for this network is ( 784 )( 16 ) + ( 16 )( 10 ) = 12 _,_ 704. The network
is implemented in an FPGA with various reuse factors from 14
to ( 784 )( 16 ) = 12 _,_ 544. In these implementations, the reduction
in DSPs can be seen as _𝑅_ is increased. Not shown in the figure is
the complementary behavior where the latency and II of the NN
increase commensurately. For example, the II of the NN increases
from 14 to 12,544 clock cycles as _𝑅_ increases. Thus, for a clock
frequency of 100 MHz, the II of the network would increase from

140 ns to 0.125 ms.


_Sparse operations._ In Sec. 3.2, pruning is presented to create more
efficient NN implementations by reducing the number of multiplication operations required to evaluate the network. By creating
a network implementation where the matrix-vector kernel has a
large fraction of zero-weights, the computation resources can be



4096


1024


256


64


16


4


DSP (HLS) DSP (Available)


**Figure 6: DSP usage for the MNIST neural network imple-**
**mentation where the reuse factor** _𝑅_ **is scanned. As** _𝑅_ **is in-**

**creased, the DSP usage decreases while the latency (not**
**shown) increases accordingly.**


greatly reduced. In hls4ml, this can be built into the NN translation through a dedicated sparse matrix-vector multiplication kernel.
There are two complementary implementations of this kernel in
hls4ml depending on the size of the matrix and the latency of the
operation required.
In the first implementation, HLS preprocessor directives are
used to limit the number of multipliers available to the kernel
based on the number of nonzero weights, and HLS is left to do the
optimization. This is only feasible for smaller network layers. In the
second implementation, the nonzero weights are compressed using
a coordinate list (COO) representation, where indices are packed
into the weights themselves. The hls4ml user can specify a Boolean
compression parameter per layer, which activates this kernel. As
an example, the COO implementation is used for the jet classifier
NN described in previous sections. The architecture is pruned using
an iterative pruning approach as described in Sec. 3.2 where the
network is pruned in increments of 10% of the original number
of network weights. Figure 7 illustrates the DSP and LUT usage
of those NNs as a function of the pruned weight percentage. The
figure shows the expected scaling of the pruned implementation
where the resources decrease as a larger percentage of the network
is pruned.


**4.3** **Device-Specific Workflows**


_4.3.1_ _Xilinx FPGA workflow with Vitis._ There are multiple ways to
execute an hls4ml project on a given FPGA. The RTL code created
by Vivado HLS is fully functional and can be placed in a Vivado
block design. While this allows the user to customize the implementation to meet specific design goals or to integrate the project
into an existing firmware design, it can also present a barrier for
less experienced developers. Vitis Accel is a Xilinx tool that aims to
assist users in accelerating FPGA kernels. A Vitis Accel design consists of two components: the FPGA kernel and the host code, which
typically runs on a CPU. While the tool supports multiple kernel
description languages, we have focused on HLS-based kernels. Vitis Accel imposes various constraints on the input and output of
the kernel that require the introduction of a wrapper around the
default hls4ml project. The host code is then able to manage the


TinyML Research Symposium’21, March 2021, San Jose, CA Fahim, et al.





















**Figure 7: DSP (top) and LUT (bottom) usage of the jet sub-**
**structure classification network as a function of the percent-**
**age of the network pruned.**


transfer data between the host CPU and the FPGA, either through
DMA transfers or AXI streams. The choice of data transfer protocol is critical to the performance of the design. Typically, a small
number of large data transfers is preferable to a large number of
small data transfers. With SoC devices there is significant flexibility in customizing the data transfers due to the many different
memory types available and their physical locations on the chip.
Vitis Accel can be used to integrate hls4ml kernels. For smaller networks run with very large batch sizes, Vitis Accel and hls4ml are
capable of producing highly performant accelerated coprocessing
systems [80].


_4.3.2_ _ASIC workflow._ Domain scientists may choose an ASIC rather
than an FPGA implementation when they aim at sacrificing reprogrammability for greater efficiency. However, designing for ASICs
is significantly more complicated and time-consuming than for FPGAs. In the ASIC design workflow, verification and power analysis
play a bigger role at the various levels of abstractions.
Figure 8 shows the ASIC workflow integrated with hls4ml . The
ML training phase provides us with both the model and the stimuli
for the subsequent verification steps. hls4ml compiles the trained
model in a synthesizable C++ specification and a set of directives
for Mentor Catapult HLS to target ASIC design [ 50 ]. Thanks to
our state-of-the-art implementation of the C++ ML specification
and optimized synthesis directives, the HLS-generated RTL code is
comparable in power, performance, and area (PPA) to handwritten
RTL [ 81 ]. ASIC design projects are often impractical for domain



**Figure 8: Design and verification stack for the ASIC work-**
**flow.**


scientists because of the hardware’s intrinsic complexity and the inability of many small-sized research groups to navigate the lengthy
RTL-based hardware design cycle to meet acceptable deployment
time frames. In our ASIC workflow, we can spend more time (1)
refining the ML model thanks to the quick PPA estimation from Catapult HLS and (2) verifying both the C++ and RTL implementations
to identify bugs and improve performance. We check design rules
on the C++ specification by performing static analysis (Mentor
CDesignChecker); we run C simulation and code coverage (Mentor
CCov); finally, we run C&RTL co-simulation for equivalence checking [ 82 ]. The synthesized RTL code is subsequently processed with
a traditional digital implementation flow that integrates simulation
steps to ensure optimal PPA.
As a recent demonstration of this workflow, a completed design
of a low-latency autoencoder for particle physics data compression
has been implemented for the low-power CMOS 65 nm technology node [ 83, 84 ]. The algorithm, trained in QKeras, compresses
on-sensor data with convolutional and dense layers to be transmitted off the detector. In order to maintain reconfigurability of the
algorithm in changing experimental conditions, the weights can be
updated via an I2C interface. The design also features triple modular redundancy to maintain its radiation tolerance up to 200 MRad.
The algorithm, which has roughly 4,400 parameters, has a latency
of 25 ns, occupies an area of 3.6 mm [2], and is estimated to consume
2.38 nJ per inference.


**5** **SUMMARY AND OUTLOOK**


In this paper, we present the current status of the open-source
codesign hls4ml workflow, focusing on new features and techniques relevant for low-power ML inference. We detail, for the
first time, the structural features of hls4ml which allow for model
introspection and validation in a Python package and support for


hls4ml: An Open-Source Codesign Workflow to Empower Scientific Low-Power Machine Learning Devices TinyML Research Symposium’21, March 2021, San Jose, CA



multiple device types. Building on previous work for quantizationaware training, we introduce quantization-aware pruning for neural
networks, which providing additional resource savings. We also
describe new hls4ml features for implementation for FPGAs and
ASICs. These include configuration handles for quantization and
pruning as well as for parallelization that allow targeting either low
latency or low power implementations. The main advantage over
similar approaches is the acceleration of the codesign process using
an all-in-one, turn-key workflow for a variety of ML models and
devices. This is especially powerful for domain scientists where
ML FPGA development can be drastically sped up from months
and years to weeks. Furthermore, a unique aspect of hls4ml is
the support for multiple vendor backends (e.g. Xilinx, Intel, and
Mentor) with possible expansion to others.
While the current features of hls4ml presented in this paper
offer a set of powerful capabilities, the ultimate goal is to provide
a complete end-to-end toolkit to empower domain scientists to
design machine learning algorithms for low-power devices. This
includes development based on dedicated domain-specific data
sets, models, platforms, and existing implemented designs for a
range of devices. Further maturation of introspection tools and
workflows for design performance, validation, and close integration
with power estimation into standard CAD tools will give neural
network designers timely feedback about the power consumption of
their design choices without the need to consult hardware experts.
Effective design-space exploration from a hardware perspective
allows domain scientists to optimize better the power-performance
trade-offs of their systems. We hope that hls4ml will lead to broader
adoption of machine learning techniques in the low-power regime
in science, enhancing scientific research with tinyML.


**ACKNOWLEDGMENTS**


F. F., B. H., C. H., J. H., N. T. are supported by Fermi Research
Alliance, LLC under Contract No. DE-AC02-07CH11359 with the
U.S. Department of Energy (DOE), Office of Science, Office of High
Energy Physics. J. D. is supported by the DOE, Office of Science, Office of High Energy Physics Early Career Research program under
Award No. DE-SC0021187. G. D. and L. C. are supported in part by
DARPA (C#: FA8650-18-2-7862) and in part by the National Science
Foundation (A#: 1764000). The views and conclusions contained
herein are those of the authors and should not be interpreted as
necessarily representing the official policies or endorsements, either expressed or implied, of Air Force Research Laboratory (AFRL)
and Defense Advanced Research Projects Agency (DARPA) or the
U.S. Government. T. A, H. J., V. L., M. P., A. A. P., and S. S. are
supported by the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation program
(Grant Agreement No. 772369).


**REFERENCES**


[1] C. R. Banbury, et al . 2020. Benchmarking TinyML Systems: Challenges and
[Direction. (2020). arXiv:2003.04821](https://arxiv.org/abs/2003.04821)

[2] S. Zhang, et al . 2020. NeckSense: A multi-sensor necklace for detecting eating
activities in free-living conditions. _Proc. ACM Interact. Mob. Wearable Ubiquitous_
_Technol._ [4 (2020), 1. https://doi.org/10.1145/3397313 arXiv:1911.07179](https://doi.org/10.1145/3397313)

[3] A. R. Elias, et al . 2017. Where’s the bear? - Automating wildlife image processing using IoT and edge cloud systems. In _2017 IEEE/ACM Second International_
_Conference on Internet-of-Things Design and Implementation (IoTDI)_ . IEEE, 247.




[4] D. Vasisht, et al . 2017. Farmbeats: An IoT platform for data-driven agriculture. In
_14th USENIX Symposium on Networked Systems Design and Implementation (NSDI_
_17)_ . 515.

[5] J. Duarte et al . 2018. Fast inference of deep neural networks in FPGAs for
particle physics. _J. Instrum._ 13 (2018), P07027. [https://doi.org/10.1088/1748-](https://doi.org/10.1088/1748-0221/13/07/P07027)
[0221/13/07/P07027 arXiv:1804.06913](https://doi.org/10.1088/1748-0221/13/07/P07027)

[6] V. Loncar et al . [2021. fastmachinelearning/hls4ml: bartsia (v0.5.0). https://github.](https://github.com/fastmachinelearning/hls4ml)
[com/fastmachinelearning/hls4ml. https://doi.org/10.5281/zenodo.4585796](https://github.com/fastmachinelearning/hls4ml)

[7] R. Nane, et al . 2016. A survey and evaluation of FPGA high-level synthesis
tools. _IEEE Trans. Comput.-Aided Des. Integr. Circuits Syst._ 35, 10 (2016), 1591.
[https://doi.org/10.1109/TCAD.2015.2513673](https://doi.org/10.1109/TCAD.2015.2513673)

[8] S. Lahti, et al . 2019. Are we there yet? A study on the state of high-level synthesis.
_IEEE Trans. Comput.-Aided Des. Integr. Circuits Syst._ [38 (2019), 898. https://doi.](https://doi.org/10.1109/TCAD.2018.2834439)
[org/10.1109/TCAD.2018.2834439](https://doi.org/10.1109/TCAD.2018.2834439)

[9] J. Ngadiuba, et al . 2020. Compressing deep neural networks on FPGAs to binary
and ternary precision with hls4ml . _Mach. Learn.: Sci. Technol._ 2 (2020), 015001.
[https://doi.org/10.1088/2632-2153/aba042 arXiv:2003.06308](https://doi.org/10.1088/2632-2153/aba042)

[10] S. Summers et al . 2020. Fast inference of boosted decision trees in FPGAs for
particle physics. _J. Instrum._ 15 (2020), P05026. [https://doi.org/10.1088/1748-](https://doi.org/10.1088/1748-0221/15/05/P05026)
[0221/15/05/P05026 arXiv:2002.02534](https://doi.org/10.1088/1748-0221/15/05/P05026)

[11] C. N. Coelho, et al . 2020. Automatic deep heterogeneous quantization of deep
neural networks for ultra low-area, low-latency inference on the edge at particle
[colliders. (2020). arXiv:2006.10159 Submitted to](https://arxiv.org/abs/2006.10159) _Nat. Mach. Intell._

[12] T. Aarrestad, et al . 2021. Fast convolutional neural networks on FPGAs with
[hls4ml. (2021). arXiv:2101.05108 Submitted to](https://arxiv.org/abs/2101.05108) _Mach. Learn.: Sci. Technol._

[13] Y. Iiyama et al . 2021. Distance-weighted graph neural networks on FPGAs for
real-time particle reconstruction in high energy physics. _Front. Big Data_ 3 (2021),
[44. https://doi.org/10.3389/fdata.2020.598927 arXiv:2008.03601](https://doi.org/10.3389/fdata.2020.598927)

[14] B. Hawks, et al . 2021. Ps and Qs: Quantization-aware pruning for efficient low
[latency neural network inference. (2021). arXiv:2102.11289 Submitted to](https://arxiv.org/abs/2102.11289) _Front._
_AI_ .

[15] I. Kuon et al . 2007. Measuring the gap between FPGAs and ASICs. _IEEE Trans._
_Comput.-Aided Design Integr. Circuits Syst._ [26 (2007), 203. https://doi.org/10.1109/](https://doi.org/10.1109/TCAD.2006.884574)
[TCAD.2006.884574](https://doi.org/10.1109/TCAD.2006.884574)

[16] M. Horowitz. 2014. 1.1 Computing’s energy problem (and what we can do about
it). In _2014 IEEE International Solid-State Circuits Conference Digest of Technical_
_Papers (ISSCC)_ [. IEEE, 10. https://doi.org/10.1109/ISSCC.2014.6757323](https://doi.org/10.1109/ISSCC.2014.6757323)

[17] S. I. Venieris, et al . 2018. Toolflows for mapping convolutional neural networks
on FPGAs: A survey and future directions. _ACM Comput. Surv._ 51, Article 56
[(2018), 39 pages. https://doi.org/10.1145/3186332 arXiv:1803.05900](https://doi.org/10.1145/3186332)

[18] K. Guo, et al . 2019. A survey of FPGA-based neural network inference accelerators.
_ACM Trans. Reconfigurable Technol. Syst._ [12, Article 2 (2019), 26 pages. https:](https://doi.org/10.1145/3289185)
[//doi.org/10.1145/3289185 arXiv:1712.08934](https://doi.org/10.1145/3289185)

[19] A. Shawahna, et al . 2019. FPGA-based accelerators of deep learning networks
for learning and classification: A review. _IEEE Access_ 7 (2019), 7823. [https:](https://doi.org/10.1109/access.2018.2890150)
[//doi.org/10.1109/access.2018.2890150 arXiv:1901.00121](https://doi.org/10.1109/access.2018.2890150)

[20] K. Abdelouahab, et al . 2018. Accelerating CNN inference on FPGAs: A survey.
[(2018). arXiv:1806.01683](https://arxiv.org/abs/1806.01683)

[21] S. I. Venieris et al . 2017. fpgaConvNet: A Toolflow for Mapping Diverse Convolutional Neural Networks on Embedded FPGAs. In _NIPS 2017 Workshop on Machine_
_Learning on the Phone and other Consumer Devices_ [. arXiv:1711.08740](https://arxiv.org/abs/1711.08740)

[22] S. I. Venieris et al . 2017. fpgaConvNet: Automated mapping of convolutional
neural networks on FPGAs. In _Proceedings of the 2017 ACM/SIGDA International_
_Symposium on Field-Programmable Gate Arrays_ [. ACM, 291. https://doi.org/10.](https://doi.org/10.1145/3020078.3021791)
[1145/3020078.3021791](https://doi.org/10.1145/3020078.3021791)

[23] S. I. Venieris et al . 2017. Latency-driven design for FPGA-based convolutional
neural networks. In _2017 27th International Conference on Field Programmable_
_Logic and Applications (FPL)_ [. 1. https://doi.org/10.23919/FPL.2017.8056828](https://doi.org/10.23919/FPL.2017.8056828)

[24] S. I. Venieris et al . 2016. fpgaConvNet: A framework for mapping convolutional
neural networks on FPGAs. In _2016 IEEE 24th Annual International Symposium_
_on Field-Programmable Custom Computing Machines (FCCM)_ [. IEEE, 40. https:](https://doi.org/10.1109/FCCM.2016.22)
[//doi.org/10.1109/FCCM.2016.22](https://doi.org/10.1109/FCCM.2016.22)

[25] Y. Jia, et al . 2014. Caffe: Convolutional architecture for fast feature embedding. In
_Proceedings of the 22nd ACM International Conference on Multimedia_ . ACM, New
[York, NY, USA, 675. https://doi.org/10.1145/2647868.2654889 arXiv:1408.5093](https://doi.org/10.1145/2647868.2654889)

[26] Y. Guan, et al . 2017. FP-DNN: An automated framework for mapping deep neural
networks onto FPGAs with RTL-HLS hybrid templates. In _2017 IEEE 25th Annual_
_International Symposium on Field-Programmable Custom Computing Machines_
_(FCCM)_ [. 152. https://doi.org/10.1109/FCCM.2017.25](https://doi.org/10.1109/FCCM.2017.25)

[27] M. Abadi, et al . 2015. TensorFlow: Large-Scale Machine Learning on Heteroge[neous Systems. arXiv:1603.04467 https://www.tensorflow.org/](https://arxiv.org/abs/1603.04467)

[28] H. Sharma, et al . 2016. From high-level deep neural models to FPGAs. In _49th_
_Annual IEEE/ACM International Symposium on Microarchitecture (MICRO)_ . IEEE,
1.

[29] R. DiCecco, et al . 2016. Caffeinated FPGAs: FPGA framework for convolutional
neural networks. In _2016 International Conference on Field-Programmable Technol-_
_ogy (FPT)_ [. IEEE, 265. https://doi.org/10.1109/FPT.2016.7929549 arXiv:1609.09671](https://doi.org/10.1109/FPT.2016.7929549)


TinyML Research Symposium’21, March 2021, San Jose, CA Fahim, et al.




[30] V. Gokhale, et al. 2017. Snowflake: An efficient hardware accelerator for convolutional neural networks. In _2017 IEEE International Symposium on Circuits and_
_Systems (ISCAS)_ [. 1. https://doi.org/10.1109/ISCAS.2017.8050809 arXiv:1708.02579](https://doi.org/10.1109/ISCAS.2017.8050809)

[31] R. Collobert, et al . 2011. Torch7: A Matlab-like environment for machine learning.
In _BigLearn, NIPS Workshop_ .

[32] K. Majumder et al . 2019. A flexible FPGA accelerator for convolutional neural
[networks. (2019). arXiv:1912.07284](https://arxiv.org/abs/1912.07284)

[33] A. Rahman, et al . 2016. Efficient FPGA acceleration of convolutional neural
networks using logical-3D compute array. In _2016 Design, Automation Test in_
_Europe Conference Exhibition (DATE)_ . IEEE, 1393.

[[34] Xilinx. 2021. Xilinx/Vitis-AI. https://github.com/Xilinx/Vitis-AI.](https://github.com/Xilinx/Vitis-AI)

[35] M. Blott, et al . 2018. FINN-R: An End-to-End Deep-Learning Framework for
Fast Exploration of Quantized Neural Networks. _ACM Trans. Reconfigurable_
_Technol. Syst._ [11, 3, Article 16 (2018), 23 pages. https://doi.org/10.1145/3242897](https://doi.org/10.1145/3242897)
[arXiv:1809.04570](https://arxiv.org/abs/1809.04570)

[36] Y. Umuroglu, et al . 2017. FINN: A framework for fast, scalable binarized neural
network inference. In _Proceedings of the 2017 ACM/SIGDA International Sym-_
_posium on Field-Programmable Gate Arrays_ [. ACM, 65. https://doi.org/10.1145/](https://doi.org/10.1145/3020078.3021744)
[3020078.3021744 arXiv:1612.07119](https://doi.org/10.1145/3020078.3021744)

[[37] Y. Umuroglu et al. 2021. Xilinx/finn. https://github.com/Xilinx/finn.](https://github.com/Xilinx/finn)

[38] Alessandro, et al . [2020. Xilinx/brevitas: bnn_pynq-r1. https://github.com/Xilinx/](https://github.com/Xilinx/brevitas)
[brevitas. https://doi.org/10.5281/zenodo.4020996](https://github.com/Xilinx/brevitas)

[[39] Google. 2020. TensorFlow Lite. https://www.tensorflow.org/lite](https://www.tensorflow.org/lite)

[40] P. N. Whatmough, et al . 2019. FixyNN: Efficient hardware for mobile computer
vision via transfer learning. In _2nd SysML Conference_ [. 13. arXiv:1902.11128](https://arxiv.org/abs/1902.11128)
[https://mlsys.org/Conferences/2019/doc/2019/69.pdf](https://mlsys.org/Conferences/2019/doc/2019/69.pdf)

[41] P. N. Whatmough et al . [2019. ARM-software/DeepFreeze. https://github.com/](https://github.com/ARM-software/DeepFreeze)
[ARM-software/DeepFreeze.](https://github.com/ARM-software/DeepFreeze)

[42] G. B. Hacene, et al . 2020. Quantized guided pruning for efficient hardware
implementations of convolutional neural networks. In _2020 18th IEEE International_
_New Circuits and Systems Conference (NEWCAS)_ [. 206. https://doi.org/10.1109/](https://doi.org/10.1109/NEWCAS49341.2020.9159769)
[NEWCAS49341.2020.9159769 arXiv:1812.11337](https://doi.org/10.1109/NEWCAS49341.2020.9159769)

[43] S.-E. Chang, et al . 2021. Mix and Match: A novel FPGA-centric deep neural
network quantization framework. In _27th IEEE International Symposium on High-_
_Performance Computer Architecture (HPCA)_ [. arXiv:2012.04240](https://arxiv.org/abs/2012.04240)

[44] M. W. Numan, et al . 2020. Towards automatic high-level code deployment on
reconfigurable platforms: A survey of high-level synthesis tools and toolchains.
_IEEE Access_ [8 (2020), 174692. https://doi.org/10.1109/ACCESS.2020.3024098](https://doi.org/10.1109/ACCESS.2020.3024098)

[45] S. Ioffe et al . 2015. Batch normalization: Accelerating deep network training by
reducing internal covariate shift. In _32nd International Conference on Machine_
_Learning_, F. Bach et al . [(Eds.), Vol. 37. PMLR, Lille, France, 448. arXiv:1502.03167](https://arxiv.org/abs/1502.03167)
[http://proceedings.mlr.press/v37/ioffe15.html](http://proceedings.mlr.press/v37/ioffe15.html)

[[46] N. Jeffries et al. 2021. mlcommons/tiny. https://github.com/mlcommons/tiny/.](https://github.com/mlcommons/tiny/)

[47] V. Nair et al . 2010. Rectified linear units improve restricted Boltzmann machines.
In _27th International Conference on International Conference on Machine Learning_
_(ICML’10)_ . Omnipress, Madison, WI, USA, 807.

[48] X. Glorot, et al . 2011. Deep sparse rectifier neural networks. In _14th International_
_Conference on Artificial Intelligence and Statistics_, G. Gordon, et al . (Eds.), Vol. 15.
[JMLR, Fort Lauderdale, FL, USA, 315. http://proceedings.mlr.press/v15/glorot11a.](http://proceedings.mlr.press/v15/glorot11a.html)
[html](http://proceedings.mlr.press/v15/glorot11a.html)

[49] [Intel. 2020. Intel high level synthesis compiler. https://www.intel.com/content/](https://www.intel.com/content/www/us/en/software/programmable/quartus-prime/hls-compiler.html)
[www/us/en/software/programmable/quartus-prime/hls-compiler.html.](https://www.intel.com/content/www/us/en/software/programmable/quartus-prime/hls-compiler.html)

[50] [Mentor/Siemens. 2020. Catapult high-level synthesis. https://www.mentor.com/](https://www.mentor.com/hls-lp/catapult-high-level-synthesis)
[hls-lp/catapult-high-level-synthesis.](https://www.mentor.com/hls-lp/catapult-high-level-synthesis)

[51] M. Pierini, et al . 2020. hls4ml [LHC jet dataset (150 particles). https://doi.org/10.](https://doi.org/10.5281/zenodo.3602260)
[5281/zenodo.3602260](https://doi.org/10.5281/zenodo.3602260)

[52] E. Coleman, et al . 2018. The importance of calorimetry for highly-boosted
jet substructure. _J. Instrum._ 13 (2018), T01003. [https://doi.org/10.1088/1748-](https://doi.org/10.1088/1748-0221/13/01/T01003)
[0221/13/01/T01003 arXiv:1709.08705](https://doi.org/10.1088/1748-0221/13/01/T01003)

[53] Y. LeCun et al . [2010. MNIST handwritten digit database. http://yann.lecun.com/](http://yann.lecun.com/exdb/mnist/)
[exdb/mnist/](http://yann.lecun.com/exdb/mnist/)

[54] Y. Gong, et al . 2014. Compressing deep convolutional networks using vector
[quantization. (2014). arXiv:1412.6115](https://arxiv.org/abs/1412.6115)

[55] J. Wu, et al . 2016. Quantized convolutional neural networks for mobile devices.
In _2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ . 4820.
[https://doi.org/10.1109/CVPR.2016.521 arXiv:1512.06473](https://doi.org/10.1109/CVPR.2016.521)

[56] V. Vanhoucke, et al. 2011. Improving the speed of neural networks on CPUs. In
_Deep Learning and Unsupervised Feature Learning Workshop at the 25th Conference_
_on Neural Information Processing Systems_ .

[57] S. Gupta, et al . 2015. Deep Learning with Limited Numerical Precision. In _32nd_
_International Conference on Machine Learning_, F. Bach et al . (Eds.), Vol. 37. PMLR,
[Lille, France, 1737. arXiv:1502.02551 http://proceedings.mlr.press/v37/gupta15.](https://arxiv.org/abs/1502.02551)
[html](http://proceedings.mlr.press/v37/gupta15.html)

[58] S. Han, et al . 2016. Deep compression: Compressing deep neural network with
pruning, trained quantization and Huffman Coding. In _4th International Confer-_
_ence on Learning Representations_ [. arXiv:1510.00149](https://arxiv.org/abs/1510.00149)

[59] I. Hubara, et al . 2018. Quantized neural networks: Training neural networks
with low precision weights and activations. _J. Mach. Learn. Res._ 18, 187 (2018), 1.



[arXiv:1609.07061 http://jmlr.org/papers/v18/16-456.html](https://arxiv.org/abs/1609.07061)

[60] M. Courbariaux, et al . 2015. BinaryConnect: Training deep neural networks
with binary weights during propagations. In _Advances in Neural Informa-_
_tion Processing Systems_, C. Cortes, et al . (Eds.), Vol. 28. Curran Associates,
Inc., 3123. [arXiv:1511.00363 https://proceedings.neurips.cc/paper/2015/file/](https://arxiv.org/abs/1511.00363)
[3e15cc11f979ed25912dff5b0669f2cd-Paper.pdf](https://proceedings.neurips.cc/paper/2015/file/3e15cc11f979ed25912dff5b0669f2cd-Paper.pdf)

[61] I. Hubara, et al . 2016. Binarized neural networks. In _Advances in Neural Infor-_
_mation Processing Systems_, D. D. Lee, et al . (Eds.), Vol. 29. Curran Associates,
Inc., 4107. [arXiv:1602.02830 https://proceedings.neurips.cc/paper/2016/file/](https://arxiv.org/abs/1602.02830)
[d8330f857a17c53d217014ee776bfd50-Paper.pdf](https://proceedings.neurips.cc/paper/2016/file/d8330f857a17c53d217014ee776bfd50-Paper.pdf)

[62] M. Rastegari, et al . 2016. XNOR-Net: ImageNet classification using binary convolutional neural networks. In _ECCV 2016_, B. Leibe, et al . (Eds.). Springer, Cham,
[Switzerland, 525. arXiv:1603.05279](https://arxiv.org/abs/1603.05279)

[63] P. Merolla, et al . 2016. Deep neural networks are robust to weight binarization
[and other non-linear distortions. (2016). arXiv:1606.01981](https://arxiv.org/abs/1606.01981)

[64] B. Moons, et al . 2017. Minimum energy quantized neural networks. In _2017 51st_
_Asilomar Conference on Signals, Systems, and Computers_ [. 1921. https://doi.org/10.](https://doi.org/10.1109/ACSSC.2017.8335699)
[1109/ACSSC.2017.8335699 arXiv:1711.00215](https://doi.org/10.1109/ACSSC.2017.8335699)

[[65] F. Li et al. 2016. Ternary weight networks. (2016). arXiv:1605.04711](https://arxiv.org/abs/1605.04711)

[66] K. Wang, et al . 2019. HAQ: Hardware-aware automated quantization. In _IEEE/CVF_
_Conference on Computer Vision and Pattern Recognition (CVPR)_ . IEEE, 8604.
[arXiv:1811.08886](https://arxiv.org/abs/1811.08886)

[67] Z. Dong, et al . 2019. HAWQ: Hessian aware quantization of neural networks
with mixed-precision. In _2019 IEEE/CVF International Conference on Computer_
_Vision (ICCV)_ [. IEEE, 293. arXiv:1905.03696](https://arxiv.org/abs/1905.03696)

[68] Y. Cheng, et al . 2018. A Survey of Model compression and acceleration for deep
neural networks. _IEEE Signal Process. Mag._ [35 (2018), 126. https://doi.org/10.](https://doi.org/10.1109/MSP.2017.2765695)
[1109/MSP.2017.2765695 arXiv:1710.09282](https://doi.org/10.1109/MSP.2017.2765695)

[69] L. Deng, et al . 2020. Model compression and hardware acceleration for neural
networks: A comprehensive survey. _Proc. IEEE_ [108 (2020), 485. https://doi.org/](https://doi.org/10.1109/JPROC.2020.2976475)
[10.1109/JPROC.2020.2976475](https://doi.org/10.1109/JPROC.2020.2976475)

[70] T. Choudhary, et al . 2020. A comprehensive survey on model compression and
acceleration. _Artif. Intell. Rev._ [53, 7 (2020), 5113. https://doi.org/10.1007/s10462-](https://doi.org/10.1007/s10462-020-09816-7)
[020-09816-7](https://doi.org/10.1007/s10462-020-09816-7)

[71] Y. LeCun, et al . 1990. Optimal brain damage. In _Advances in Neural Information_
_Processing Systems_ [, D. S. Touretzky (Ed.), Vol. 2. Morgan-Kaufmann, 598. http:](http://papers.nips.cc/paper/250-optimal-brain-damage)
[//papers.nips.cc/paper/250-optimal-brain-damage](http://papers.nips.cc/paper/250-optimal-brain-damage)

[72] C. Louizos, et al . 2018. Learning sparse neural networks through _𝐿_ 0 regularization.
In _6th International Conference on Learning Representations_ [. arXiv:1712.01312](https://arxiv.org/abs/1712.01312)
[https://openreview.net/forum?id=H1Y8hhg0b](https://openreview.net/forum?id=H1Y8hhg0b)

[73] J. Frankle et al . 2019. The lottery ticket hypothesis: Training pruned neural networks. In _7th International Conference on Learning Representations_ .
[arXiv:1803.03635 https://openreview.net/forum?id=rJl-b3RcF7](https://arxiv.org/abs/1803.03635)

[74] A. Renda, et al . 2020. Comparing rewinding and fine-tuning in neural network pruning. In _8th International Conference on Learning Representations_ .
[arXiv:2003.02389](https://arxiv.org/abs/2003.02389)

[75] D. Blalock, et al . 2020. What is the state of neural network pruning?.
In _Proceedings of Machine Learning and Systems_, I. Dhillon, et al . (Eds.),
[Vol. 2. 129. arXiv:2003.03033 https://proceedings.mlsys.org/paper/2020/file/](https://arxiv.org/abs/2003.03033)
[d2ddea18f00665ce8623e36bd4e3c7c5-Paper.pdf](https://proceedings.mlsys.org/paper/2020/file/d2ddea18f00665ce8623e36bd4e3c7c5-Paper.pdf)

[76] S. Han, et al . 2015. Learning both weights and connections for efficient neural
networks. In _Advances in Neural Information Processing Systems_, C. Cortes, et al .
[(Eds.), Vol. 28. Curran Associates, Inc., 1135. arXiv:1506.02626 https://proceedings.](https://arxiv.org/abs/1506.02626)
[neurips.cc/paper/2015/file/ae0eb3eed39d2bcef4622b2499a05fe6-Paper.pdf](https://proceedings.neurips.cc/paper/2015/file/ae0eb3eed39d2bcef4622b2499a05fe6-Paper.pdf)

[77] A. Y. Ng. 2004. Feature selection, L1 vs. L2 regularization, and rotational invariance. In _21st International Conference On Machine Learning (ICML ’04)_ . ACM,
[New York, NY, USA, 78. https://doi.org/10.1145/1015330.1015435](https://doi.org/10.1145/1015330.1015435)

[78] C. Baskin, et al . 2018. UNIQ: Uniform noise injection for the quantization of
[neural networks. (2018). arXiv:1804.10969](https://arxiv.org/abs/1804.10969)

[79] Xilinx. 2018. _7 Series DSP48E1 slice user guide_ [. https://www.xilinx.com/support/](https://www.xilinx.com/support/documentation/user_guides/ug479_7Series_DSP48E1.pdf)
[documentation/user_guides/ug479_7Series_DSP48E1.pdf](https://www.xilinx.com/support/documentation/user_guides/ug479_7Series_DSP48E1.pdf)

[80] D. S. Rankin, et al . 2020. FPGAs-as-a-Service Toolkit (FaaST). In _2020 IEEE/ACM_
_International Workshop on Heterogeneous High-performance Reconfigurable Com-_
_puting_ [. arXiv:2010.08556](https://arxiv.org/abs/2010.08556)

[81] B. Khailany, et al . 2018. A modular digital VLSI flow for high-productivity SoC
design. In _ACM/ESDA/IEEE Design Automation Conference (DAC)_ . IEEE, 1.

[82] [Mentor/Siemens. 2020. Catapult high-level synthesis - Verification. https://www.](https://www.mentor.com/hls-lp/catapult-high-level-synthesis/hls-verification)
[mentor.com/hls-lp/catapult-high-level-synthesis/hls-verification.](https://www.mentor.com/hls-lp/catapult-high-level-synthesis/hls-verification)

[83] C. Herwig, et al . 2020. Design of a reconfigurable autoencoder algorithm for
detector front-end ASICs. In _IEEE Nuclear Science Symposium & Medical Imaging_
_Conference_ . IEEE.

[84] F. Fahim, et al . 2020. High-level synthesis to on-chip implementation of a reconfigurable AI accelerator for front-end data analysis at the HL-LHC. In _IEEE_
_Nuclear Science Symposium & Medical Imaging Conference_ . IEEE.


