# Neural Network Inference on Mobile SoCs

Siqi Wang, Anuj Pathania, Tulika Mitra



_**Abstract**_ **—The ever-increasing demand from mobile Machine**
**Learning (ML) applications calls for evermore powerful on-chip**
**computing resources. Mobile devices are empowered with hetero-**
**geneous multi-processor Systems-on-Chips (SoCs) to process ML**
**workloads such as Convolutional Neural Network (CNN) infer-**
**ence. Mobile SoCs house several different types of ML capable**
**components on-die, such as CPU, GPU, and accelerators. These**
**different components are capable of independently performing**
**inference but with very different power-performance character-**
**istics. In this article, we provide a quantitative evaluation of**
**the inference capabilities of the different components on mobile**
**SoCs. We also present insights behind their respective power-**
**performance behavior. Finally, we explore the performance limit**
**of the mobile SoCs by synergistically engaging all the components**
**concurrently. We observe that a mobile SoC provides up to 2x**
**improvement with parallel inference when all its components are**
**engaged, as opposed to engaging only one component.**
_**Index Terms**_ **—Deep learning, convolutional neural networks,**
**heterogeneous computing, embedded multiprocessor SoCs**


I. I NTRODUCTION


The tremendous popularity of Neural-Network (NN) based
machine learning applications in recent years has been fuelled
partly by the increased capability of the compute engines, in
particular, the GPUs. Traditionally, both the network training
and inference were performed on the cloud with mobile
devices only acting as user interfaces. However, enriched user
experience and privacy concerns now demand inference to
be performed on the mobile devices themselves with high
accuracy and throughput.
In this article, we look at NN-enabled vision applications on mobile devices. These applications extract highlevel semantic information from real-time video streams and

predominately use Convolutional Neural Networks (CNNs).
They are important in many domains, such as Advanced
Driver-Assistance Systems (ADAS), Virtual Reality (VR), and
Augmented Reality (AR). Enabling these applications in the
power-constrained mobile devices is challenging due to the
enormous computational and memory requirements.
Heterogeneous multi-processor SoC enables the current
state-of-the-art mobile devices. However, the presence of
multiple vendors fragments the mobile SoCs. Accelerators
(including GPU, FPGA, and dedicated neural accelerators)
demonstrate great performance for inference. However, these
high-performance components are present in only a small
fraction of the mobile devices. Moreover, due to market
fragmentation, it is impossible to develop a mobile application


S. Wang, A. Pathania and T. Mitra are with the Department of Computer Science, School of Computing, National University of Singapore. Email: ((wangsq, pathania, tulika)@comp.nus.edu.sg). Address: COM1, 13
Computing Drive, S117417. (Corresponding author: Tulika Mitra)


|Col1|Col2|
|---|---|
|Small CPU|DVFS|
|Core<br>Core<br>Core<br>Core<br>L2 Cache|Core<br>Core<br>Core<br>Core<br>L2 Cache|



CCI Bus


DRAM


Fig. 1: An abstract block diagram of a mobile SoC with an
asymmetric multi-core CPU, GPU, and NPU.


with accelerators that can run across multiple devices. Instead,
the CPUs remain the common denominator among mobile
SoCs and is the favored choice for inference [1].
We embark on an exploration to quantitatively characterize
and understand the inferencing capabilities of the mobile
SoCs given the diverse landscape. We portray the powerperformance gap between the ubiquitous CPUs and the highperformance accelerators in high-end devices and uncover the
reasons behind the gap through the roofline models. Finally,
we propose simultaneous engagement of all the SoC components to greatly expand the promise of functional deployment
of vision applications on mobile devices.


II. I NFERENCE ON M OBILE S O C S


_A. Heterogeneous Multi-processor SoCs_


There are over two thousand unique mobile SoCs in the
mobile devices market. The diversity comes from the choice
of different CPUs, GPUs, caches, memory controllers, and
other application-specific accelerators. This fragmentation of
the SoC market makes standard optimizations impossible.
However, the similarity among these SoCs lies in the choice
of one or more CPU core clusters.

_1) ARM big.LITTLE:_ Multi-cores enable the state-of-theart Mobile SoCs. 99.9% of the _Android_ devices in the market

in 2019 have multiple cores [1]. Among these, about half
of the SoCs implement performance heterogeneity with at
least two CPU clusters: a high-performance and an energyefficient core cluster. _ARM big.LITTLE_ architecture, one of the
most popular architectures implementing this heterogeneity, is
present in _Hi-Silicon Kirin_, _Samsung Exynos_, and _Qualcomm_
_Snapdragon_ series SoCs. The heterogeneous cores differ in
power-performance-area characteristics but share the same
Instruction Set Architecture (ISA). Figure 1 shows an abstract


















block diagram of this architecture. The general availability of
CPUs make them a favorable choice for mobile inference and

make device-agnostic optimizations feasible.
_2) Accelerators:_ Existing architectures, including GPU and
FPGA, have proven to be advantageous for ML workloads and
are thus commonly used for deployment on certain devices.
Both academic and commercial dedicated accelerators ( _Google_
_Edge TPU_, _Intel Nervana NNP_, _Huawei NPU_, _Apple Neural_
_Engine_ ) offer exceptional runtime and energy-efficiency. There
are no standard neural accelerators for mobile SoCs, making
horizontal application integration difficult. Limited availability
even constraints the use of GPUs.


_B. Mobile ML Framework and Optimizations_


_Tensorflow_, _PyTorch_, and _MXNet_ are some of the common
ML development frameworks for all scenarios. _Tensorflow Lite_
like frameworks facilitates the compression of huge models to
fit into resource-constrained mobile devices. Efficient libraries
and APIs bridge the gap between the frameworks and the
underlying hardware, examples of which are _Nvidia cuDNN_
for GPUs, _ARM NN_ powered by _Compute Library (ARM-_
_CL)_ for ARM CPUs and GPUs, _Facebook NNPACK_, and
_QNNPACK_ for mobile CPUs. These libraries usually optimize
with detailed architectural information. _ARM-CL_ supports acceleration through _ARM_ NEON vectorization and provides
NEON assembly implementation for the most computationally intensive convolution kernels. Algorithmic optimizations
(Winograd transform, FFT, sparsity exploration) lower the
computational complexity of convolution computations. Furthermore, quantization and network pruning are common
techniques that bring down the processing requirement with
the sacrifice of accuracy [2].
Even though most mobile inference workloads run on CPUs,
optimizations of ML workloads with accelerators hordes most
of the attention. There is a lot of room for optimizations
on mobile CPUs to enable ML applications across different
mobile platforms.


III. C HARACTERIZING I NFERENCING ON M OBILE S O C


We perform experiments across different technology nodes
using two commonly used mobile SoCs: 28 nm _Exynos 5422_
within _Odroid XU3_ development platform and 10 nm _Kirin_
_970_ within _Hikey 970_ development platform. Released in 2014
and 2017 respectively, these two SoCs show us the progress of
mobile SoCs development over the years. Furthermore, these
two SoCs roughly approximate the mid- and high-end mobile
SoCs today.
In the experiments, both SoCs are using _ARM-CL 18.05v_ .
_Kirin 970_ NPU is supported by _HiAI DDK (v100)_ for network
deployment. For _Exynos5422_, in-built power sensors, running
at 200 Hz, measure the power of each component. For _Kirin_
_970_, because of the absence of any integrated on-chip power
sensors, we approximate the power consumption by measuring
the socket power with the help of a power measurement
unit [3] running at 100 Hz.



TABLE I: Throughput of different networks on different
mobile SoCs components running at their peak frequencies.








|Network|Exynos 5422<br>Throughput (Imgs/s)|Col3|Col4|Kirin 970<br>Throughput (Imgs/s)|Col6|Col7|Col8|
|---|---|---|---|---|---|---|---|
|Network|_A7_|_A15_|_T628_|_A53_|_A73_|_G72_|NPU|
|_AlexNet_|1.1|3.1|7.8|2.2|7.6|32.5|32.5|
|_GoogLeNet_|0.9|3.4|5.2|3.0|7.1|19.9|34.4|
|_MobileNet_|1.5|5.7|8.5|6.5|17.7|29.1|_Not Supported_|
|_ResNet50_|0.2|1.3|2.1|1.5|2.8|8.4|21.9|
|_SqueezeNet_|1.5|5.0|8.0|6.8|15.7|43.0|49.3|



_A. Experimental Set-up_


_1) CPU:_ Both SoCs include _ARM big.LITTLE_ based asymmetric multi-core CPU. _Kirin 970_ CPU adopts _ARMv8-A_
architecture. It consists of a high-performance high-power
out-of-order four-core _Cortex-A73_ cluster (2.36 GHz) and
a low-performance low-power four-core in-order _Cortex-_
_A53_ (1.8 GHz). _Exynos 5422_ has a similar design but uses
an older _ARMv7-A_ architecture with _Cortex-A15_ (2 GHz) and
_Cortex-A7_ (1.4 GHz) cores. All CPU cores support NEON
advanced Single Instruction Multiple Data (SIMD) operations,
which allows for four 32-bit floating-point operations per
cycle.
_2) GPU: Kirin 970_ adopts _ARM Mali G72 MP12 GPU_
(850 MHz), implementing the second generation _Bifrost_ architecture. It has twelve shader cores with three execution

engines each. Each engine is capable of eight FP32 operations per cycle, giving a total peak compute capability of
244.8 GFLOPS/s for _G72_ . _Exynos 5422_ includes an _ARM_
_Mali T628 MP6_ GPU (600 MHz). It adopts an older _Midgard_
architecture with six shader cores implementing _Tripipe_ design
with two arithmetic pipelines. Each pipeline is capable of eight
FP32 operations per cycle, providing a total peak compute
capability of 57.6 GFLOPS/s for _T628_ .
_3) NPU: Kirin 970_ includes a _Huawei_ NPU purposebuilt for ML. It has a peak performance of 1.92 TFLOPS/s
with FP16. The accompanying _HiAi DDK_ API enables the
deployment of networks on NPU but only works with _Android_ .
_Exynos 5422_ does not have any ML accelerator.
_4) Network_ _Structure:_ We experiment with several
popular networks introduced in recent years – _AlexNet_ [4],
_GoogleNet_ [5], _MobileNet_ [6], _ResNet50_ [7], and
_SqueezeNet_ [8].


_B. Individual Heterogeneous Components_


We first study each component in isolation by running inferencing of multiple images in a stream on a single component.
Both _Big_ and _Small_ clusters are self-sufficient for inferencing.
GPU and NPU require the support of a _Small_ cluster for
inferencing.
_1) Throughput:_ Table I shows the throughput of each component on both our SoCs. All components in _Kirin 970_ outperform their respective counterparts in older _Exynos 5422_ . _Big_
_A73_ cluster, _Small A53_ cluster, and _G72_ GPU outperform _Big_
_A15_ cluster, _Small A7 cluster_, and _T628_ GPU on average by a
factor of 4.4x, 2.6x, and 4.2x, respectively. The performance
gap between the _Big_ and _Small_ cluster has reduced from 4x to


Fig. 2: Energy efficiency of different components while running at their peak frequencies.


2.5x with a decrease in _Big_ to _Small_ power consumption ratio
from 10x to 4x. Furthermore, the performance gap between
GPU and CPU clusters is only about 2x to 3x for both SoCs.
For NPU, we were unable to deploy _MobileNet_ due to
incompatible operators. On average, NPU is only 1.6x better
than the high-end _G72_ GPU. On the other hand, the portability
of applications across different platforms remains a challenge
for dedicated accelerators. The proprietary development kit
makes the general optimization a difficult endeavor.
_2) Energy Efficiency:_ We measure the average active power
consumption of inferencing on different components and calculate the energy efficiency, as shown in Figure 2. For _Exynos_
_5422_, power sensors for individual components measure the
power consumption of each component separately. For _Kirin_
_970_, we calculate active power values by subtracting the idle
power (measured when no workload is running) from socket
power measurement taken during inferencing. Therefore, the
power measurements for _Kirin_ are slightly higher, as memory
power cannot be separated.
NPU is the most energy-efficient among all components,
which we expect, given its custom design for inference. GPUs
are the second-most energy-efficient component. _Small_ clusters
also show good energy-efficiency. However, Table I shows
their performance in terms of absolute throughput is too low
to be ever useful alone.

Comparing across two platforms, the energy efficiency of
each component has improved for the newer SoC. However,
the improvement is minimal and even negative for the _Small_
CPU cluster. Compared to its predecessor A7, A53 is more
complex and area hungry with 64-bit, complex branch prediction, and larger TLB. It achieves greater performance but at
the cost of even greater power consumption.
_3) Impact of Technology Scaling Versus Architectural In-_
_novations: Exynos 5422_ and _Kirin 970_ use the 28 nm and
10 nm technology nodes, respectively. In moving from 28 nm
_Exynos 5422_ to 10 nm _Kirin 970_, the maximum frequency
of the _Big_ cluster has only changed from 2 GHz ( _A15_ ) to
2.36 GHz ( _A73_ ), while the _Small_ cluster changes from 1.4 GHz
( _A7_ ) to 1.8 GHz ( _A53_ ). So the frequency scaling is 1.18x for
the big cluster and 1.29x for the _Small_ cluster for these two



platforms. On the other hand, we get 4.4x and 2.6x throughput
improvement across technology generations (Table I) for _Big_
cluster and _Small_ cluster, respectively. This improvement in
performance is achieved through smart designs such as microarchitectural improvements (improved branch predictor, cache
data prefetchers, etc.), larger caches, and 64-bit support leading to improved NEON processing, among others.
However, in the case of the small cluster, with an increased
area, the micro-architectural changes give an increase in power
that cannot be offset by technology scaling. Indeed, the small
_A53_ cluster consumes roughly twice the power of the small
_A7_ cluster. Thus, the energy-efficiency improvement is limited
for the small cluster for some networks as we move from

_A7_ to _A53_ . In contrast, between the two big clusters, _A73_ is
more power-efficient compared to _A15_ ; the energy-efficiency
improves from _A15_ to _A73_ cluster. As mentioned earlier, the
power measurements for _A7_ and _A15_ are quite accurate, while
the measured power for _A53_ and _A73_ are higher as it includes
the memory power that could not be separated.
_4) Insights:_ We observe that NPU provides unmatched
energy-efficiency for inferences. It is the optimal choice to
perform network inferences on the platforms with such dedicated accelerators. However, a developer needs to put in
substantial effort to port their application with proprietary
API to execute on NPU, and the effort would not bear any
fruits on mobile devices lacking this very-specific NPU. NPU,
as a black-box, also causes inflexibility in development and
optimizations. Furthermore, NPU is compatible with only a
limited set of network designs. These extra requirements could
make it quickly obsolete for future networks.
On the other hand, high-end GPUs can provide performance
comparable to NPU at satisfactory energy-efficiency. GPUs
are capable of running General-Purpose (GPGPU) applications
written in _OpenCL_, which is easily portable to a large variety
of GPUs and even CPUs supporting _OpenCL_ . This generality
makes it a good candidate to use when high performance is a
major consideration.
CPUs provide both the worst energy-efficiency as well as
the worst throughput among all components. Still, they are
critical for inferencing because they are commonly present
across all mobile devices. Low-end mobile SoCs would lack

accelerators like NPU. They may contain a low-end GPU,
but maybe missing _OpenCL_ support and thereby lack any
inferencing capability. Network inference on CPU is inevitable
and demands optimization considerations.
Our analysis shows that any component alone on both
platforms can barely support the increasing performance requirement for network inferencing. Section V-A presents the
co-execution methodology that can mitigate the performance
issue to some extent. Still, we must continue to look into
the networks themselves in search of further optimization
opportunities.


IV. R OOFLINE A NALYSIS


To understand the execution behaviors of the networks

on each SoC components, we perform a roofline analysis.


Roofline analysis [9] is a widely applied methodology that can
classify an application as memory- or compute-bound on given
hardware. It gives insights to developers for improving their
application design to cater to the computation and memory
capability of the underlying processing devices. The horizontal
“Ceiling” and the “Roof” constructs a “Roofline” that bounds
the maximum performance of an application (measured in
GOPS/s) under a hardware-determined compute- or memorybound, respectively. Operational Intensity (OI) of application (measured in FLOPS/byte) determines whether its peak
performance is bounded by the memory bandwidth (measured
in GB/s) or compute capability (measured in GOP/s) of the
hardware. Both _Exynos 5422_ and _Kirin 970_ show similar
behavior for the CPU core clusters and GPU. Therefore, we
only present here the analysis for _Exynos 5422_ .


_A. Construction of a Roofline Model_


Hardware specifications provide the peak pure compute
performance. Micro-benchmarking [10] provides the peak
(sustainable) memory bandwidth. Specifications claim peak
memory bandwidth of the memory bus to be 14.9 GB/s. However, we observe the actual component-wise peak bandwidth
to be 3.44 GB/s, 0.49 GB/s, and 6.15 GB/s for _A15_ cluster, _A7_
cluster, and _T628_ GPU, respectively.
Many variations of the roofline model are constructed to
adapt to different use-cases. In this analysis, we defined
two operational intensities, that are, theoretical OI ( _OI_ _t_ ) and
empirical OI ( _OI_ _e_ ), defined in Eqn (1) and (2).


_OI_ _t_ = _GOPS/Mem Access_ (1)


_OI_ _e_ = _GOPS/DRAM Access_ (2)


We calculate _OI_ _t_ by analyzing the code. The memory accesses
include all the data required in the computation. During
actual executions, multiple levels of caches within components
improve the memory access performance. The caches make it
difficult for _OI_ _t_ to correlate with the actual performance on
the components. Therefore, we introduce empirical operational
intensity _OI_ _e_ . We calculate _OI_ _e_ using the actual DRAM
accesses on the bus, which models the presence of multi-level
memory hierarchy. It is more informative and has a better
correlation with the actual performance on the component
than _OI_ _t_ . We use application-specific performance counters
obtained from _ARM Streamline DS5_ at run-time for calculation

of _OI_ _e_ (CPU: _L2 data refill_, GPU: _Mali L2 cache external_
_read/write bytes_ ). Fig. 3(a) show the roofline points of major
layers in _AlexNet_ on _A15_ cluster for both _OI_ _t_ and _OI_ _e_ .


_B. Theoretical and Empirical OI_


Figure 3(a) plots the _OI_ _t_ (squares) and _OI_ _e_ (diamonds)
values of several _AlexNet_ major layers, marked with different
colors. Black marks the whole network _OI_ _t_ and _OI_ _e_ of
AlexNet. The intersection points of the _OI_ _t_ values with the
“Roofline” represent the theoretical maximum performance for
the code-based theoretical operational intensities, which fall in
the memory-bound region on the “Roof”. The corresponding



points for _OI_ _e_ are actual achieved performance in GOPS/s,
which are always below the “Roofline”.
The presence of cache reduces the memory accesses going
to the DRAM during execution, and thus increases the operational intensity. Therefore, for all layers, _OI_ _e_ points are on the
right of _OI_ _t_ points, indicating higher performance. For layers
with low _OI_ _t_ (fully connected, FC), the points move along the
“Roofline”, achieving the theoretical maximum performance.
For layers with higher _OI_ _t_ (convolutional, CONV), the points
cross the boundary of memory-bound and become computebound. The performance gain is not as significant, and we
explain this with the underutilization due to insufficient or
imperfect parallelization. Overall, _OI_ _e_ is a better indicator of
real-world performance. Therefore, we only plot values of _OI_ _e_
going forward.


_C. Across Different Components_


Figure 3(b) shows the performance of different networks on
different components on _Exynos 5422_ . The color of the points
corresponds to the respective component. We can observe
that memory severely bottlenecks the performance of both _A7_
cluster and _T628_ GPU. Performance of _A15_ cluster falls in

both compute- and memory-bound regions depending upon
the network.

The _OI_ _e_ values are different because of the different memory hierarchies for different components. The _Big_ core cluster
with a larger cache size (L2: 2MB) derives higher benefits
from memory hierarchy than GPU (L2: 128KB). However,
_AlexNet_ that is notorious for huge parameter sizes caches
will get flushed regardless of the cache sizes resulting in a
smaller benefit from the memory hierarchy. On the other hand,
small filter sizes lead to sub-optimal parallelization (underutilization). This observation holds more starkly for newer
networks with smaller filter size than older networks. The
observation explains the significant deviation in the empirical performance of networks on the components from the
“Roofline”.


_D. Major Layers in Inference_


We do a deeper layer-level analysis to explain the behavior
of the networks. Both convolutional and fully-connected layers
dominate the total execution time of networks, and thus both
are considered as major layers worthy of examination. We
limit our analysis to _Big_ cluster because networks there show
both memory- and compute-bound behavior. Figure 3(c) shows
that different layers in _AlexNet_ (and also other networks to a
lesser extent) exhibits different empirical OIs. Convolutional
layers at the start of _AlexNet_ perform compute-intensive convolution on large inputs and thereby have relatively higher OIs.
On the other hand, fully-connected layers perform memoryintensive operations on large size parameters and thereby
have relatively lower OIs. Convolutional and fully-connected
layers of _AlexNet_ fall in the compute- and memory-bound
region of the roofline model, respectively. Overall, _AlexNet_
falls somewhere in the middle of both.


AlexNet_OI_t AlexNet_OI_e CONV1_OI_t
CONV1_OI_e CONV3_OI_t CONV3_OI_e
FC1_OI_t FC1_OI_e FC3_OI_t
FC3_OI_e A15 roofline

100


10


1


0.1


0.01

0.01 0.1 1 10


Operational Intensity (OPS/Byte)


(a) Roofline plot with theoretical (OI_t) and
emperical (OI_e) operational intensities for
AlexNet (black) and some major layers
(colors) on Exynos 5422 A15 CPU cluster.



A15 roofline A7 roofline

Mali-T628 roofline AlexNet
GoogLeNet MobileNet
ResNet50 SqueezeNet


20


2


0.2

0.5 5


Operational Intensity (OPS/Byte)


(b) Comparison of different processor
roofline with emperical operational
intensities for five CNN applications on
Exynos 5422 A15, A7 and GPU.



A15 roofline AlexNet

GoogLeNet MobileNet
ResNet50 SqueezeNet


10


1


0.1

0.05 0.5 5


Operational Intensity (OPS/Byte)


(c) Roofline plot with major layer
information for five CNN applications on
Exynos 5422 A15 CPU cluster.



Fig. 3: Roofline plot for inference workloads and major layer information on multiple processors in _Exynos 5422_ .



In general, we observe that layers of a network are scattered
in both compute- or memory-bound region. This difference
comes from the choice of the size of the input tensors and
filters. The vast differences in _OI_ _e_ for different layers within a
network motivates layer-level optimizations such as per-layer
Dynamic Voltage and Frequency Scaling (DVFS) for power
management. Furthermore, the variation within a network
motivates fine-grain layer level co-executions, which improve
the overall chip utilization [11].


_E. Effect of Quantization_


Quantization is a commonly applied technique that reduces
the memory and computation requirement of a network while
reducing accuracy. However, the quality of its implementation primarily determines the benefits it provides. In the
implementation of quantized MobileNet in _ARM-CL_ (18.05v),
QASYMM8 model with 8-bit weights is used. This implementation fails to improve the overall performance of the network.
Deeper analysis reveals that the latencies of convolutional
layers are indeed reduced, but the overheads from extensive
de-quantization and re-quantization overshadow any benefit.
Quantization reduces the total operations and memory
access required near-proportionally. Reduction in memory
accesses results in a slightly higher empirical operational
intensity _OI_ _e_ . Therefore, the roofline analysis of a quantized
network nearly overlaps with that of its non-quantized counterpart, and quantization does not improve the memory behavior
of the layers. Lower operation requirements under quantization
predominately contribute to the reduction in execution time of
the convolutional layers.


_F. Glimpse of NPU_


NPU, due to its novelty and dedicated machine learning
processing design, garners a lot of attention. However, most
of the details are kept confidential. We are unaware of its
architectural and integration details. Therefore, we can only
attempt to reverse engineer its behavior to gain some insights.



TABLE II: Throughput improvement on _Exynos 5422_ and
_Hikey 970_ by co-execution over the best throughput with a



|ingle compo|onent (T628 and G72 GP|Col3|Col4|PU).|Col6|
|---|---|---|---|---|---|
|Network|_Exynos 5422_<br>Throughput (Imgs/s)|_Exynos 5422_<br>Throughput (Imgs/s)|_Exynos 5422_<br>Throughput (Imgs/s)|_Kirin 970_<br>Throughput (Imgs/s)|_Kirin 970_<br>Throughput (Imgs/s)|
|Network|_T628_|Co-<br>execution|Gain|_G72_<br>Co<br>execut|-<br>ion<br>Gain|
|_AlexNet_|7.8|10.3|32.4%|32.5<br>3|3.4<br>2.8%|
|_GoogLeNet_|5.2|8.7|66.3%|19.9<br>2|8.4<br>42.8%|
|_MobileNet_|8.5|14.9|76.7%|29.1<br>5|1.5<br>77.1%|
|_ResNet50_|2.1|2.9|38.6%|8.4<br>1|2.3<br>46.3%|
|_SqueezeNet_|8.0|13.8|73.9%|43.0<br>5|4.5<br>26.7%|


We implement a kernel module that enables counting of
traffic on the CCI bus. We attribute the traffic on the CCI
bus that goes to DRAM during the engagement of NPU to
the main memory activity of NPU. The maximum observed
memory bandwidth of executing several networks and the peak
performance of 1.92 TOPS from the specification construct the
“Roof” and “Ceiling” of the NPU roofline. We observe that the
performance of NPU is significantly bounded by the memory
for the networks tested. This observation shows a significant
scope for optimization to achieve the full processing potential
of NPU.


V. I MPROVING THE PERFORMANCE


_A. Co-Execution of Multiple Components_


Stream processing, depending on the application, requires
10 to 40 images/second throughput. Some applications even
require multiple inferences to run at the same time. Table I
shows that the high-end _Kirin 970_ SoC can barely sustain
such requirement while the mid-end _Exynos 5422_ cannot.
We previously observed that peak bandwidth consumed by
any individual component is far below the total bandwidth
supported by the bus. This observation supports the claim
that inferencing through multiple components together will not
make individual components more memory-constrained compared to their isolated inferencing. Therefore, we use _ARM-_








TABLE III: Throughput improvement on _Kirin 970_ by coexecution over the best throughput with a single component (NPU).









|Network|Throughput (Images/s)|Col3|Gain<br>(%)|Image Frames<br>Composition (%)|Col6|Col7|Col8|
|---|---|---|---|---|---|---|---|
|Network|NPU|Co-<br>execution|Co-<br>execution|_A73_|_A53_|_G72_|NPU|
|_AlexNet_|32.5|63.7|96.0|1.90|0.95|47.47|49.68|
|_GoogleNet_|34.4|59.3|72.4|3.06|1.70|33.33|61.90|
|_ResNet50_|21.9|30.9|40.9|2.63|1.32|26.97|69.08|
|_SqueezeNet_|49.3|95.1|92.9|3.18|1.69|43.43|51.69|


Fig. 4: Energy efficiency of co-execution on _Exynos 5422_ with
all components, on _Kirin 970_ with CPU and GPU (excluding
NPU) and all components (including NPU).


_CL_ to create an infrastructure, wherein multiple components
process images from a single unified stream in parallel using
a work-stealing mechanism. The infrastructure uses a buffer
to reorder the out-of-sync output from different components.
Co-execution obtains significantly higher throughput than the
highest throughput component in isolated execution.
Table II shows the peak co-execution throughput on both
mobile SoCs with the _ARM big.LITTLE_ CPU core cluster and
GPU. We include the best individual component executions,
which are GPU for both platforms, for comparison. On average, the co-execution gives 50% throughput improvement
over GPU only execution. Furthermore, Table II shows _Exynos_
_5422_ ’s obsolescence. Even with the co-execution, _Exynos 5422_
shows very low absolute throughput.


_B. Co-execution with NPU_


The performance of NPU is unbeatable. Table III shows
that _Kirin 970_, with co-execution of all on-chip components,
gives exceptionally high throughput. In practice, we can
execute NPU and GPU in parallel towards one application
that demands very high performance or to perform multiple
inferences simultaneously with multiple applications.


_C. Co-Execution Energy Efficiency_


Synergistic co-execution engages multiple components simultaneously to improve performance at the cost of higher
power consumption. Therefore, the energy efficiency of the
co-execution is the average energy efficiency of engaged
components. Figure 4 shows the energy efficiency of the
execution that engages all the components on _Exynos 5422_,
the CPU clusters and GPU on _Kirin 970_ (exclude NPU), and
all the components on _Kirin 970_ (include NPU). Overall, the
co-execution energy efficiency is always better than the _Big_
CPU cluster. In _Kirin 970_ SoC, as GPU is much more energyefficient than the CPU clusters, the co-execution provides



better energy efficiency than the power-efficient _Small_ CPU
cluster.


VI. S UMMARY


Mobile inferencing is now ubiquitous. In this work, we
examine the power-performance characteristics of inferencing through several prominent neural networks on different
components available within a mobile SoC. We also perform
roofline analysis of networks on components to unveil the
further optimization scope. We show that network throughput
can increase by up to 2x using co-execution that engages all
the components in inferencing simultaneously.


**Siqi Wang** is currently a research assistant and is working toward the Ph.D.
degree at School of Computing, National University of Singapore. Her current
research interests include performance optimization, task scheduling, general
purpose GPUs and deep learning on heterogeneous multi-processor systems.


**Anuj Pathania** is currently working as a research fellow at School of
Computing, National University of Singapore. He received his Ph.D. degree
from Karlsruhe Institute of Technology (KIT), Germany in 2018. His research
focuses on resource management algorithms with emphasis on performance-,
power- and thermal-efficiency in embedded systems.


**Tulika Mitra** is a Professor of Computer Science at School of Computing,
National University of Singapore. She received her PhD degrees in computer
science from the State University of New York Stony Brook in 2000. Her
research interests span various aspects of the design automation of embedded
real-time systems, cyber-physical systems, and Internet of Things.


R EFERENCES


[1] C.-J. Wu, D. Brooks, K. Chen, D. Chen, S. Choudhury, M. Dukhan,
K. Hazelwood, E. Isaac, Y. Jia, B. Jia _et al._, “Machine learning at
facebook: Understanding inference at the edge,” in _2019 IEEE Interna-_
_tional Symposium on High Performance Computer Architecture (HPCA)_ .
IEEE, 2019, pp. 331–344.

[2] M. Wess, S. M. P. Dinakarrao, and A. Jantsch, “Weighted quantizationregularization in dnns for weight memory minimization toward hw
implementation,” _IEEE Transactions on Computer-Aided Design of_
_Integrated Circuits and Systems_, vol. 37, no. 11, pp. 2929–2939, 2018.

[3] “Keysight Technologies B2900 Series Precision Source/Measure Unit,”
[https://goo.gl/U4HMbu.](https://goo.gl/U4HMbu)

[4] A. Krizhevsky, I. Sutskever, and G. E. Hinton, “Imagenet classification
with deep convolutional neural networks,” in _Advances in neural infor-_
_mation processing systems_, 2012, pp. 1097–1105.

[5] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan,
V. Vanhoucke, and A. Rabinovich, “Going deeper with convolutions,”
in _Proceedings of the IEEE conference on computer vision and pattern_
_recognition_, 2015, pp. 1–9.

[6] A. G. Howard, M. Zhu, B. Chen, D. Kalenichenko, W. Wang, T. Weyand,
M. Andreetto, and H. Adam, “Mobilenets: Efficient convolutional neural
networks for mobile vision applications,” _arXiv preprint:1704.04861_,
2017.

[7] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image
recognition,” in _Proceedings of the IEEE conference on computer vision_
_and pattern recognition_, 2016, pp. 770–778.

[8] F. N. Iandola, S. Han, M. W. Moskewicz, K. Ashraf, W. J. Dally,
and K. Keutzer, “SqueezeNet: AlexNet-level accuracy with 50x fewer
parameters and 0.5 MB model size,” _arXiv preprint :1602.07360_, 2016.

[9] S. Williams, A. Waterman, and D. Patterson, “Roofline: An insightful
visual performance model for floating-point programs and multicore
architectures,” Lawrence Berkeley National Lab.(LBNL), Berkeley, CA
(United States), Tech. Rep., 2009.

[[10] S. Siamashka, “Tinymembench,” https://github.com/ssvb/tinymembench.](https://github.com/ssvb/tinymembench)

[11] S. Wang, G. Ananthanarayanan, Y. Zeng, N. Goel, A. Pathania,
and T. Mitra, “High-throughput cnn inference on embedded arm
big.little multi-core processors,” _IEEE Transactions on Computer-Aided_
_Design of Integrated Circuits and Systems_, 2019. [Online]. Available:
[http://dx.doi.org/10.1109/TCAD.2019.2944584](http://dx.doi.org/10.1109/TCAD.2019.2944584)


