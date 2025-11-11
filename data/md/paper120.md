## I MPROVEMENT IN LAND COVER AND CROP CLASSIFICATION BASED ON TEMPORAL FEATURES LEARNING FROM S ENTINEL -2 DATA USING R ECURRENT -C ONVOLUTIONAL N EURAL N ETWORK (R-CNN)



**Vittorio Mazzia**
Department of Electronics and Telecommunications
Politecnico di Torino
10124 Turin, Italy
```
    vittorio.mazzia@polito.it

```


**Aleem Khaliq**
Department of Electronics and Telecommunications
Politecnico di Torino
10124 Turin, Italy
```
     aleem.khaliq@polito.it

```


**Marcello Chiaberge**
Department of Electronics and Telecommunications
Politecnico di Torino
10124 Turin, Italy
```
                marcello.chiaberge@polito.it

```

May 6, 2020


**A** **BSTRACT**


Understanding the use of current land cover, along with monitoring change over time, is vital for
agronomists and agricultural agencies responsible for land management. The increasing spatial and
temporal resolution of globally available satellite images, such as provided by Sentinel-2, creates
new possibilities for researchers to use freely available multi-spectral optical images, with decametric
spatial resolution and more frequent revisits for remote sensing applications such as land cover and
crop classification (LC&CC), agricultural monitoring and management, environment monitoring.
Existing solutions dedicated to cropland mapping can be categorized based on per-pixel based and
object-based. However, it is still challenging when more classes of agricultural crops are considered
at massive scale. In this paper, a novel and optimal deep learning model for pixel-based LC&CC
is developed and implemented based on Recurrent Neural Networks (RNN) in combination with
Convolutional Neural Networks (CNN) using multi temporal sentinel-2 imagery of central north part
of Italy, which has diverse agricultural system dominated by economic crop types. The proposed
methodology is capable of automated features extraction by learning time correlation of multiple
images, which reduces manual feature engineering and modeling crops phenological stages. Fifteen
classes, including major agricultural crops, were considered in this study. We also tested other
widely used traditional machine learning algorithms for comparison such as support vector machine
SVM, random forest (RF), Kernal SVM, and gradient boosting machine, also called XGBoost. The
overall accuracy achieved by our proposed Pixel R-CNN was 96.5%, which showed considerable
improvements in comparison with existing mainstream methods. This study showed that Pixel R-CNN
based model offers a highly accurate way to assess and employ time-series data for multi-temporal
classification tasks.


_**K**_ **eywords** satellite imagery _·_ deep Learning _·_ pixel-based crops classification _·_ recurrent neural networks _·_
convolutional neural networks.


P IXEL R-CNN


**1** **Introduction**


Significant increase in population around the globe, demanding increase in agricultural productivity and thus precise
land cover and crop classification and spatial distribution of various crops are becoming significant for governments, policymakers and farmers to improve decision-making process to manage agricultural practices and needs

[ Gomez(2016) ]. Crop maps are produced relatively at large scale, ranging from global[ Wu(2014) ], countrywide

[ Jin(2019) ], and local level[ Yang(2007), Wang(2016) ]. The growing need for agriculture in the management of
sustainable natural resources becomes essential for the development of effective cropland mapping and monitoring

[ Matton(2015) ]. Group on Earth Observations (GEO), with its Integrated Global Observing Strategy (IGOS), also
emphases on an operational system for monitoring global land covers and mapping spatial distribution of crops by
using remote sensing imagery. spatial information of the crop maps have been the main source for crop growth
monitoring[ Huang(2015), Battude(2016), Guan(2016) ], water resources managment [ Toureiro(2017) ], decsion making
for policy makers to ensure food security [Yu(2017)].
Satellite and Geographic Information System (GIS) data have been an important source factor in establishing and
improving the current systems that are responsible for developing and maintaining land cover and agricultural maps

[ Boryan(2011) ]. Freely available satellite data offers one of the most applied sources for mapping agricultural land
and assessing important indices that describe conditions of crop fields [ Xie(2008) ]. Recently launched sentinel-2 is
equipped with multispectral imager that can provide up to 10m per pixel spatial resolution with the revisit time of 5
days, which offers great opportunity to be exploited in remote sensing domain.
Multispectral time series data acquired from MODIS and LANDSAT have been widely used in many agricultural
applications such as crop yield prediction [ Johnson(2016) ], landcover and crop classification [ Senf(2015) ], leaf area
index estimation [ Jin(2016) ], plant height estimation [ Liaqat(2017) ], vegetation variability assessment [ Senf(2017) ]
and many more. Two different data sources can also be used together to extract more features that lead to improving
results. For example, Landsat-8 and sentinel-1 used together for LC&CC [Kussul(2016)].
There are some supervised or unsupervised algorithms for mapping cropland using mono or multi-temporal images

[ Xiong(2017), Yan(2015) ]. Multi-temporal images have already proven to gain better performance than mono temporal
mapping methods [ Gomez(2016), Xiao(2018) ]. The imagery used for only key phenological stages s proved to be
sufficient for crop area estimation [ Gallego(2008), Khaliq(2018) ]. It has also found in [ Zhou(2013) ], that reducing
time-series length affects the average accuracy of the classifier. Crop patterns were established using Enhanced
Vegetation Index derived from 250 meters MODIS-Terra time series data and used to classify some major crops like
corn, cotton, and soybean in Brazil [ Arvor(2011) ]. Centimetric resolution imagery is available at the cost of high price
of commercial satellite imagery or with the extensive UAV flight campaigns to cover large area during the whole crop
cycle to get better spatial and temporal details. However, most of the studies used moderate spatial resolution(10-30m)
freely available satellite imagery for land cover mapping due to their high spectral and temporal resolution which is
difficult in case of UAV and high resolution satellite imagery.
Other than multispectral time series data, several vegetation indices (VIs) derived from different spectral bands have been
exploited and used to enrich the feature space for vegetation assessment and monitoring [ Zhong(2012), Wardlow(2012) ].
VIs such as normalized difference vegetation index (NDVI), normalized difference water index (NDWI), enhanced
vegetation indexes (EVI), textural features, such as grey level co-occurrence matrix (GLCM), statistical features, such as
mean, standard deviation, inertial moment are the features more frequently used for crop classification. It is possible to
increase the accuracy of the algorithms also using ancillary data such as elevation, census data, road density, or coverage.
Nevertheless, all these derived features, along with the phenological metrics involve a huge volume of data, which may
increase computational complexity with little improvement in accuracy [ Löw(2013) ]. Several feature selection methods
have been proposed [ Löw(2013) ] to deal with this problem. In [ Hao(2015) ], various features have been derived from
MODIS time series and best feature selection has been made using random forest algorithm.
LC&CC can also be classified as pixel-based or object-based. Object-based image analysis (OBIA), described by
Blaschke, that segmentation of satellite images into homogeneous image segments can be achieved with high-resolution
sensors [ Blaschke(2010) ]. Various object-based classification has been proposed to produce crop maps using satellite
imagery [Novelli(2016), Long(2013), Li(2015)].
In this work, we proposed a unique deep neural network architecture for LC&CC, which comprises of Recurrent
Neural Network (RNN) that extracts temporal correlations from time series of sentinel-2 data in combination with
Convolutional Neural Network (CNN) that analyzes and encapsulate the crops pattern through its filters. The remainder
of this paper is organized as follows. Section II briefs about related work done for the LC&CC along with an overview
of RNN and CNN. Section III provides an overview of the raw data collected and exploited during the research. Section
IV provides detailed information on the proposed model and the training strategies. Section V contains a complete
description of the experiments, results, and discussion along with the comparison with previous state-of-the-art results.
Finally, Section VI draws some conclusions.


2


P IXEL R-CNN


**2** **Related work**


**2.1** **Temporal feature representation**


There are various studies proposed in the past to address LC&CC. A more common approach adopted for classification
tasks is to extract temporal features and phenological metrics from the VIs time series derived from remotely sensed
imagery. There are also some simple statistics and threshold-based procedures used to calculate vegetation related
metrics such as Maximum VI and time of peak VI [ Walker(2014), Walker(2015) ], which have improved classification
accuracy when compared to using only VI as features [ Simonneaux(2008) ]. More complex methods have been adapted
to extract temporal features and patterns to address the vegetation phenology [ Shao(2016) ]. Further, time series of VI
represented by a set of functions [ Galford(2008) ], linear regression [ Funk(2009) ], Markov model [ Siachalou(2015) ]
and curve-fitting functions. Sigmoid function has been exploited by [ Xin(2015), Xin(2016) ], and achieved better
results due to its robustness and ease to derive phenological features for the characterization of vegetation variability

[ Dannenberg(2015) ]. Although above-mentioned methods of temporal feature extraction offer many alternatives and
flexibilities in deployment to assess vegetation dynamics, in practice, there are some important factors such as manually
designed model and feature extraction, intra-class variability, uncertain atmospheric conditions, empirical seasonal
patterns, which make the selection of such methods more difficult. Thus, an appropriate approach is needed to fully
utilize the sequential information from time series of VI to extract temporal patterns. As our proposed DNN architecture
is based on pixel classification, therefore the following subsections will provide relevant studies and description


**2.2** **Pixel based crops classification**


A detailed review of the state-of-the-art supervised pixel-based methods for land cover mapping was performed in

[ Khatami(2016) ]. It was found that support vector machine (SVM) for mono temporal image classification was the
most efficient in terms of overall accuracy (OA) of about 75%. The second approach was neural networks (NN) based
classifier with almost the same OA 74%. SVM is complex and resource-consuming for time series multispectral
data applications with broad area classification. Another common approach in the remote sensing applications is the
random forest (RF)-based classifiers [ Gislason(2006) ]. Nevertheless, multiple features should be derived to feed the RF
classifier for effective use. Deep Learning (DL) is a branch of machine learning, and it is a powerful tool that is being
widely used in solving a wide range of problems related to signal processing, computer vision, image processing, image
understanding, and natural language processing [ LeCun(2015) ]. The main idea is to discover not only the mapping
from representation to output but also the representation itself. That is achieved by breaking a complex problem into a
series of simple mappings, each described by a different layer of the model, and then composing them in a hierarchical
fashion. A large number of state of the art models, frameworks, architecture, and benchmark databases of reference
imagery exist for image classification domain.


**2.3** **Recurrent neural network (RNN)**


Sequence data analysis is an important aspect in many domains, ranging from natural language processing, handwriting
recognition, image captioning, to robot automation. In recent years, Recurrent Neural Networks (RNNs) have proven
to be a fundamental tool for sequence learning [ Sutskever(2014) ], allowing to represent information from context
window of hundreds of elements. Moreover, the research community has, over the years, come up with different
techniques to overcome the difficulty of training over many time steps. For example, long short-term memory (LSTM)

[ Sutskever(2000) ] and gated recurrent unit (GRU) [ Chung(2014) ], based architectures have proven ground-breaking
achievements [ Chung(2015), Bahdanau(2014) ], in comparison to standard RNN models. In remote sensing applications,
RNNs are commonly used when sequential data analysis is needed. For example, Lyu et al. [ Lyu(2016) ] employed
RNN to use sequential properties such as spectral correlation and intra-bands variability of multispectral data. They
further used LSTM model to learn a combined spectral-temporal feature representation from an image pair acquired at
two different dates for change detection [Lyu(2018)]


**2.4** **Convolutional neural network (CNN)**


Convolutional Neural Networks (CNNs) date back decades [ LeCun(1998) ], emerging from the study of the brain’s
visual cortex [ Hubel(1962) ] and classical concepts of computer vision theory [ Hubel(2010) ], [ Hubel(2015) ]. Since
the 1990s, these have been applied successfully in image classification [ LeCun(1998) ]. However, due to technical
constraints such as mainly lack of hardware performance, the large amount of data, and theoretical limitations, CNNs
did not scale to large applications. Nevertheless, Geoffrey Hinton and his team demonstrated at the annual ImageNet
ILSVRC [ Krizhevsky(2012) ] competition the feasibility to train large architectures capable of learning several layers
of features with increasingly abstract internal representations [ Zeiler(2014) ]. Since that breakthrough achievement,


3


P IXEL R-CNN


CNNs became the ultimate symbol of the Deep Learning [ LeCun(2015) ] revolution, incarnating all those concepts that
underpin the entire novel movement.
In recent years, DL was widely used in data mining and remote sensing applications. In particular, image classification
studies exploited several DL architectures due to their flexibility in feature representation, and automation capability
for end-to-end learning. In DL models, features can be automatically extracted for classification tasks without feature
crafting algorithms by integrating autoencoders [ Wan(2017), Mou(2017) ]. 2D CNNs have been broadly used in remote
sensing studies to extract spatial features from high-resolution images for object detection and image segmentation

[ Kampffmeyer(2016), Kampffmeyer(2017), Audebert(2018) ]. In crop classification, 2D convolution in the spatial
domain performed better than 1D convolution in the spectral-domain [ Kussul(2017) ]. These studies formed multiple
convolutional layers to extract spatial and spectral features from remotely sensed imagery.


**3** **Study area and data**


The study site near Carpi, Emilia-Romagna, situated in center-north part of Italy with central
coordinates 44 _[◦]_ 47 _[′]_ 01 _[′′]_ _N,_ 10 _[◦]_ 59 _[′]_ 37 _[′′]_ _E_ was considered for LC& CC shown in Figure 1. The Emilia-Romagna
region is one of the most fertile plains of Italy. An area almost 2640 _km_ [2] was considered, which covers diverse crop
land. The major crop fields in this region are Maize, Lucerne, Barley, Wheat, and Vineyards. The yearly averaged
temperature and precipitation are 14 _[◦]_ _C_ and 843 _mm_ for this region. Most of the farmers practice single cropping in
this area.


Table 1: Bands used in this study.



Bands used Description Central
wavelength
( _µ_ m)



Resolution
(m)



Band 2 Blue 0.49 10

Band 3 Green 0.56 10

Band 4 Red 0.665 10

Band 8 Near infrared 0.705 10

NDVI (Band8-Band4)/ (Band8+Band4)  - 10


To know about the spatial distribution of crops, we deeply studied Land Use Cover Area frame statistical Survey
(LUCAS) and extracted all the information we need for ground truth data. LUCAS was carried out by Eurostat to be able
to monitor the agriculture, climate change, biodiversity, forest, and water for almost all over the Europe [ Kussul(2017) ].


The technical reference document of LUCAS-2015 was used to prepare the ground truth data. Microdata that contains
spatial information of crops and several land cover types along with the geo-coordinates for the considered region was
imported in Quantum Geographic information system (QGIS) software, an Open-source software used for visualization,
editing, analysis of geographical data. The selection of pixel was made manually by overlapping images and LUCAS
data, so a proper amount of ground truth pixels were extracted for training and testing the algorithm. The sentinel-2
mission consists of twin polar-orbiting satellites launched by European Space Agency (ESA) in 2015 and can be used
in various application areas such as land cover change detection, natural disaster monitoring, forest monitoring, and
most importantly in agricultural monitoring and management [Kussul(2017)].


It is equipped with multi-spectral optical sensors that capture 13 bands of different wavelengths. We used only
high-resolution bands that have 10 meter/pixel resolution shown in Table 1. It also has high revisit time ( ten days at
the equator and five days with twin satellites (Sentinel-2A, Sentinel-2B). It became more popular in remote sensing
community due to fact that it possesses various key features such as, free access to data products available at ESA
Sentinel Scientific Data Hub with reasonable spatial resolution (which is 10m for Red, Green, Blue and Near Infrared
bands), high revisit time and reasonable spectral resolution among other available free data sources. In our study,
we used ten multitemporal sentinel-2 images reported in Table. 2, which are well co-registered from July-2015 to
July-2016 with close to zero cloud coverage. The initial image selection was performed based on the cloudy pixel
contribution at the granule level. This pre-screening was followed by further visual inspection of scenes and resulted in
a multi-temporal layer stack of ten images. Sentinel Application Platform (SNAP) v5.0 along with sen2core v 2.5.1
were used to apply radiometric and geometric corrections to acquire Bottom of Atmosphere (BOA) Level 2A images
from Top of Atmosphere (TOA) Level 1C. Further details about geometric, radiometric correction algorithms used in
sen2cor can be found in [ Kaufman(1988) ]. Bands with 10 meter/pixel along with the derived Normalized Difference
Vegetation Index (NDVI) were used for experiments, as shown in Table 1.


4


P IXEL R-CNN


Figure 1: The study site is located in Carpi, region Emilia-Romagna is shown with the geo-coordinates (WGS84). RGB
image composite derived from sentinel-2 imagery acquired in August-2015 is shown and the yellow marker showing
geo-locations of ground truth land cover extracted from Land Use and Coverage Area frame Survey (LUCAS-2015).


Table 2: Sentinel-2 data acquisition.


Date Doy Sensing Orbit # Cloud pixel
percentage

7/4/2015 185 22-Descending 0
8/3/2015 215 22-Descending 0.384
9/2/2015 245 22-Descending 4.795
9/12/2015 255 22-Descending 7.397
10/22/2015 295 22-Descending 7.606
2/19/2016 50 22-Descending 5.8
3/20/2016 80 22-Descending 19.866
4/29/2016 120 22-Descending 18.61
6/18/2016 170 22-Descending 15.52
7/18/2016 200 22-Descending 0


5


P IXEL R-CNN


Figure 2: Few examples of zoomed in part of crop classes considered as ground truth. Shape files are used to extract
pixels for reference data.


**4** **Convolutional and recurrent neural networks for pixel-based crops classification**


**4.1** **Formulation**


A single multi-temporal, multi-spectral pixel can be represented as a two-dimensional matrix _X_ [(] _[i]_ [)] _∈_ R _[t][∗][b]_ where _t_
and _b_ are the number of time steps and spectral bands, respectively. Our goal is to compute from _X_ [(] _[i]_ [)] a probability
distribution _F_ ( _X_ [(] _[i]_ [)] ) consisting of _K_ probabilities, where _K_ is equal to the number of classes. In order to achieve this
objective, we propose a compact representation learning architecture composed of three main building blocks:


_•_ **Time correlation representations**   - this operation extracts temporal correlations from multi-spectral, temporal
pixels _X_ [(] _[i]_ [)] exploiting a sequence-to-sequence recurrent neural network based on Long Short-Term Memory
(LSTM) cells. A final Time-Distributed layer is used to compress and maintain a sequence like structure,
preserving the multidimensionality nature of the data. In this way, it is possible to take advantage of temporal
and spectral correlations simultaneously.


_•_ **Temporal pattern extraction**   - this operation consists of a series of convolutional operations followed by
rectifier activation functions that non linearly maps each elaborated temporal and spectral patterns onto high
dimensional representations. So, RNN output temporal sequences are processed by a subsequent cascade of
filters, which in a hierarchical fashion, extracts essential features for the successive stage.


_•_ **Multiclass classification**   - this final operation maps the feature space with a probability distribution _F_ ( _X_ [(] _[i]_ [)] )
with _K_ different probabilities, where _K_, as previously stated, is equal to the number of classes.


The comprehensive pipeline of operations constitutes a lightweight, compact architecture able to non-linearly map
multi-temporal information with its intrinsic nature, achieving results considerably better than previous state-ofthe-art solutions. Human brain mental imagery studies [ Mellet(1996) ], where images are a form of internal neural
representation, inspired the presented architecture. Moreover, the joint effort of RNN and CNN distributes the knowledge
representation through the entire model, exploiting one of the most powerful characteristics of deep learning known
as distributed learning. An overview of the overall model, dubbed Pixel R-CNN, is depicted in Fig. 3. Each pixel is
extracted contemporary from all images taken at different time steps _t_ with all its spectral bands _b_ . In this way, it is
possible to create an instance _X_ [(] _[i]_ [)], which can feed the first layer of the network. Firstly, the model extracts temporal
representations from the input sample. Subsequently, these temporal features are further enriched by the convolutional
layers that patterns in a hierarchical manner. The overall model act as a function _F_ ( _X_ [(] _[i]_ [)] ) that map the input sample
with its related probabilities _K_ . So, evaluating the probability distribution is possible to identify the class of belonging
to the input sample.


It is worth to notice that this model is known as unrolled through time representation. Indeed, only after all time steps
have been processed, the CNN is able to analyze and transform the temporal pattern. In the following subsections, we
are going to describe in detail each individual block.


6


P IXEL R-CNN


Figure 3: An overview of the Pixel R-CNN model used for classification. Given a multi-temporal, multi-spectral input
pixel _X_ [(] _[i]_ [)], the first layer of LSTM units extracts sequences of temporal patterns. A stack of convolutional layers
hierarchically processes the temporal information.


**4.1.1** **Time correlation representation**


Nowadays, a popular strategy in time series data analysis is the use of RNNs that have proven excellent results in many
fields of the application over the years. Looking at the simplest possible RNN shown in Fig. 4, composed of just one
layer, it looks very similar to a feedforward neural network, except it also has a connection going backward. Indeed, the
layer is not only fed by an input vector _x_ [(] _[i]_ [)], but it also receives _h_ [(] _[i]_ [)] (cell state), which is equal to the output neuron
itself, _y_ [(] _[i]_ [)] . So, at each time step _t_, this recurrent layer receives an input 1-D array _x_ [(] _t_ _[i]_ [)] as well as its own output from
the previous time step, _y_ ( [(] _t_ _[i]_ _−_ [)] 1) [. In general, since the output of a recurrent neuron at time step] _[ t]_ [ is a function of all inputs]
from previous time steps, it has, intuitively, a sort of memory that influences all successive outputs. In this example, it is
straightforward to compute a cell’s output, as shown in Eq. 1.


_y_ _t_ [(] _[i]_ [)] = _φ_ ( _x_ [(] _t_ _[i]_ [)] _· W_ _x_ + _y_ ( [(] _t_ _[i]_ _−_ [)] 1) _[·][ W]_ _[y]_ [ +] _[ b]_ [)] (1)


Figure 4: A recurrent layer and its unrolled through time representation. A multi-temporal, multi-spectral pixel _X_ [(] _[i]_ [)] is
made by a sequence of time steps, _x_ [(] _t_ _[i]_ [)] [, that along the previous output] _[ h]_ [(] _[i]_ [)] [ feed the next iteration of the network.]


7


P IXEL R-CNN


Figure 5: LSTM with peephole connections. A time step _t_ of a multi-spectral pixel _x_ [(] _t_ _[i]_ [)] is processed by the memory
cell which decides what to add and forgot in the long-term state _c_ ( _t_ ) and what discard for the present state _y_ _t_ [(] _[i]_ [)] [.]


where, in the context of this research, _x_ [(] _t_ _[i]_ [)] _∈_ R [(1] _[∗][b]_ [)] is a single time step of a pixel with _n_ _inputs_ equal to the number of
spectral bands b. _y_ _t_ [(] _[i]_ [)] and _y_ ( [(] _t_ _[i]_ _−_ [)] 1) [are the output of the layer at time] _[ t]_ [ and] _[ t][ −]_ [1] [, respectively,] _[ W]_ _[x]_ [ and] _[ W]_ _[y]_ [ are the weights]

matrices. It is important to point out that _y_ _t_ as _x_ [(] _t_ _[i]_ [)] are vectors and they can have an arbitrary number of elements, but
the representation Fig. 4 does not change. Simply, all neurons are hidden in the depth dimension. Unfortunately, the
basic cell just described suffer from major limitations, but most of all are the fact that, during training, the gradient
of the loss function gradually fades away. For this reason, for the time correlation representation, we adopted a more
elaborated cell known as peephole LSTM unit, see Fig. 5. That is an improved variation of the concept proposed in
1997 by Sepp Hochreiter and Jurgen Schmidhuber [ Sutskever(2000) ]. The key idea is that the network can learn what
to store in a long-term state, _c_ ( _t_ ) what to throw away and what to use for the current state _h_ ( _t_ ) and _y_ ( _t_ ) that, as for the
basic unit, are equal. That is performed with simple element-wise multiplications working as ” _valves_ ” for the fluxes
of information. Those elements, _V_ 1, _V_ 2 and _V_ 3 are controlled by fully connected (FC) layers that have as input the
current input state _x_ ( _t_ ) and the previous short-term memory term _h_ ( _t−_ 1) . Moreover, for the peephole LSTM cell, the
previous long-term state _c_ ( _t−_ 1) is added as an input to the FC of the forgot gate, _V_ 1, and the input gate, _V_ 2 . Finally, the
current long-term state _c_ _t_ is added as an input to the FC of the output gate. All "gates controllers" have sigmoid as
activation functions (green boxes) instead of tanh ones to process the signals themselves (red boxes). So, to summarize,
a peephole LSTM block has three signals as input and output; two are the standard input state _x_ ( _t_ ) and cell output
_y_ ( _t_ ) . Instead, _c_ and _h_ are the long and short-term state, respectively, that the unit, utilizing its internal controllers and
valves, can feed with useful information. Formally, as for the basic cell seen before, Eq (2). Eq (7). summarizes how to
compute the cell’s long-term state, its short-term state, and its output at each time step for a single instance.


_i_ ( _t_ ) = _σ_ ( _W_ _ci_ _[T]_ _[·][ c]_ ( _t−_ 1) [+] _[ W]_ _[ T]_ _hi_ _[·][ h]_ ( _t−_ 1) [+] _[ W]_ _[ T]_ _xi_ _[·][ x]_ [(] ( _[i]_ _t_ [)] ) [+] _[ b]_ _[i]_ [)] (2)

_f_ ( _t_ ) = _σ_ ( _W_ _cf_ _[T]_ _[·][ c]_ ( _t−_ 1) [+] _[ W]_ _[ T]_ _hf_ _[·][ h]_ ( _t−_ 1) [+] _[ W]_ _[ T]_ _xf_ _[·][ x]_ [(] ( _[i]_ _t_ [)] ) [+] _[ b]_ _[f]_ [)] (3)

_o_ ( _t_ ) = _σ_ ( _W_ _co_ _[T]_ _[·][ c]_ ( _t_ ) [+] _[ W]_ _[ T]_ _ho_ _[·][ h]_ ( _t−_ 1) [+] _[ W]_ _[ T]_ _xo_ _[·][ x]_ [(] ( _[i]_ _t_ [)] ) [+] _[ b]_ _[o]_ [)] (4)

_g_ ( _t_ ) = tanh( _W_ _hg_ _[T]_ _[·][ h]_ ( _t−_ 1) [+] _[ W]_ _[ T]_ _xg_ _[·][ x]_ [(] ( _[i]_ _t_ [)] ) [+] _[ b]_ _[g]_ [)] (5)

_c_ ( _t_ ) = _f_ ( _t_ ) _⊗_ _c_ ( _t−_ 1) + _i_ ( _t_ ) _⊗_ _g_ ( _t_ ) (6)
_y_ ( _t_ ) = _h_ ( _t_ ) = _o_ ( _t_ ) _⊗_ tanh( _c_ ( _t_ ) ) (7)


In conclusion, multi-temporal, multi-spectral pixel _X_ [(] _[i]_ [)] is processed by a first layer of LSTM peephole cells obtaining
a cumulative output _Y_ ( [(] _lstm_ _[i]_ [)] ) [. Finally, a TimeDistributedDense layer is applied which executes simply a Dense function]


8


P IXEL R-CNN



**LSTM**


_Y_ _[(i)]_

(lstm)



_y_ _0_ _[(i)]_



_y_ _1_ _[(i)]_



_y_ _t_ _[(i)]_



LSTM LSTM







_x_ _0_ _[(i)]_



_x_ _1_ _[(i)]_


**Input**

_X_ _[(i)]_



_x_ _t_ _[(i)]_



Figure 6: Pixel R-CNN first layer for time correlations extraction. Peephole LSTM cells extracts temporal representations from input instances _X_ [(] _[i]_ [)] _∈_ R _[t][∗][b]_ . The output matrix _Y_ ( [(] _lstm_ _[i]_ [)] ) [feeds a TimeDistributedDense layer, that preserves]
the multidimensional nature of the processed data extracting multi-spectral patterns.


across every output over time, using the same set of weights, preserving the multidimensional nature of the processed
data Eq.(8). In Fig. 6 is presented a graphical representation of the first layer of the network. LSTM cells extract
temporal representations from input samples _X_ [(] _[i]_ [)] with _x_ [(] _t_ _[i]_ [)] _∈_ R [(1] _[∗][b]_ [)] as columns. The output matrix _Y_ ( [(] _lstm_ _[i]_ [)] ) [feeds the]
subsequent TimeDistributedDense layer.


_F_ _timeD_ ( _F_ _lstm_ ( _X_ [(] _[i]_ [)] )) = ( _W · Y_ ( [(] _lstm_ _[i]_ [)] ) [+] _[ B]_ [)] (8)


**4.1.2** **Temporal patterns extraction**


The first set of layers extract a 2-dimensional tensor _Y_ ( [(] _timeD_ _[i]_ [)] ) [for each instance. In the second operation, after a simple]
reshaping operation in order to increase the dimensionality of the input tensor from two to three and beeing able to
apply the following operations, we map each of these 3-dimensional array **Y** [(] ( _[i]_ _reshape_ [)] ) [into an higher dimensional space.]
That is accomplished by two convolutional operations, built on top of each other, that hierarchically apply learned
filters, extracting gradually more abstract representations. More formally, the temporal patterns extraction is expressed,
for example, for the first convolutional layer, as an operation _F_ _conv_ 1


_F_ _conv_ 1 ( _F_ _timeD_ ( _F_ _lstm_ ( _X_ [(] _[i]_ [)] ))) = max(0 _, W_ 1 _∗_ **Y** [(] ( _[i]_ _reshape_ [)] ) [+] _[ B]_ [1] [)] (9)


where _W_ 1 and _B_ 1 represent filters and biases, respectively, and ’ _∗_ ’ is the convolutional operation. _W_ 1, contains _n_ 1
filters with kernel dimension _f_ 1 x _f_ 1 x _c_, where _f_ 1 is the spatial size of a filter and _c_ is the number of input channels.
As common for CNN, rectified linear unit (ReLU), max(0,x), has been chosen as activation function for both layers
units. In Fig. 7 is depicted a graphical scheme of this section of the model. So, summarizing, output matrix _Y_ ( [(] _timeD_ _[i]_ [)] )


9


P IXEL R-CNN







**Conv2D**


**Y** [(i)]

(conv2)


**Reshape**

**Y** [(i)]

(reshape)



_n_ _2_


_n_ _1_


_1_



_f_ _1_



Figure 7: Pixel R-CNN convolutional layers. Firstly, output from TimeDistributedDense layer _Y_ ( [(] _timeD_ _[i]_ [)] ) [is reshaped in a]

3-dimensional tensor **Y** [(] ( _[i]_ _reshape_ [)] ) [and then it feeds a stack of two convolutional layers that progressively reduce the first]
two dimensions, gradually extracting higher level represenetations.


of the TimeDistributedDense layer, after adding an extra dimension, feeds a stack of two convolutional networks that
progressively reduce the first two dimensions, gradually extracting higher level representations and generating high
dimensional arrays. Moreover, being the _n_ 1 and _n_ 2 filters shared across all units, the same operation carried out with a
similarly-sized dense fully connected layers would require a much greater number of parameters and computational
power. Instead, the synergy of RNN and CNN opens the possibility to elaborate the overall temporal canvas in an
optimize and efficient way.


**4.1.3** **Multiclass classification**


In the last stage of the network, the extracted feature tensor **Y** [(] ( _[i]_ _conv_ [)] 2) [, after removing the extra dimensions with a simple]
flatten operation, is mapped to a probability distribution consisting of _K_ probabilities, where _K_ is equal to the number
of classes. This is achieved by a weighted sum followed by a softmax activation function:


ˆ exp _s_ _k_ ( _x_ )
_p_ _k_ = _σ_ ( _s_ ( _x_ )) _k_ = ~~�~~ _Kj_ =1 [exp] _[ s]_ _[j]_ [(] _[x]_ [)] _for j_ = 1 _.., K_ (10)


where _s_ ( _x_ ) = _W_ _[T]_ . _y_ ( [(] _flatten_ _[i]_ [)] _−conv_ 2) [+] _[ B]_ [ is a vector containing the scores of each class for the input vector]

_y_ ( [(] _flatten_ _[i]_ [)] _−conv_ 2) [, that after the flatten operaton is a 1-dimensional array. Weights W and bias B are learned, dur-]
ing the training process, in such a way to classify arrays of the high dimensional space into the K different classes. So,

ˆ
_p_ _k_ is the estimated probability that the extracted feature vector _y_ ( [(] _flatten_ _[i]_ [)] _−conv_ 2) [belongs to class k given the scores of]
each class for that instance.


10


P IXEL R-CNN


**4.2** **Training**


Learning the overall mapping function F requires the estimation of all network parameters Θ of the three different
model parts. This is simply achieved through minimizing the loss between each pixel class prediction _F_ ( _X_ [(] _[i]_ [)] ) and the
corresponding ground truth _y_ [(] _[i]_ [)] with a supervised learning strategy. So, given a data set with _n_ pixel samples _{X_ _i_ _}_ and
the respective true classes set _{y_ _i_ _}_, we use categorical cross-entropy as the loss function:



_K_
� _y_ _k_ [(] _[i]_ [)] [log(ˆ] _[p]_ [(] _k_ _[i]_ [)] [)] (11)


_k_ =1



_J_ (Θ) = _−_ 1 _/n_



_n_
�


_i_ =1



where _y_ _k_ [(] _[i]_ [)] cancels all classes loss except for the true one. Equation (11). is minimized using AMSGrad optimizer

[ Reddi(2019) ], an adaptive learning rate method which modifies the basic ADAM optimizer [ Kingma(2019) ] algorithm.
The overall algorithm update rule without the debiasing step is:


_m_ _t_ = _β_ 1 _m_ _t−_ 1 + (1 _−_ _β_ 1 ) _g_ _t_ (12)


_v_ _t_ = _β_ 2 _v_ _t−_ 1 + (1 _−_ _β_ 2 ) _g_ _t_ [2] (13)


ˆ
_v_ _t_ = max(ˆ _v_ _t−_ 1 _, v_ _t_ ) (14)


_η_
_θ_ _t_ +1 = _θ_ _t_ _−_ ˆ _t_ (15)
~~_√_~~ _v_ _t_ + _ϵm_


Equation (12). and Eq. (13). are the exponential decay of the gradient and gradient squared, respectively. Instead, with
the Eq. (14)., keeping a higher _v_ _t_ term results in a much lower learning rate, _η_, fixing the exponential moving average
and preventing to converge to a sub-optimal point of the cost function. Moreover, we use a technique known as cosine
aneling in order to cyclically vary the learning rate value between certain boundary values [ Smith(2017) ]. This value
can be obtained with a preliminary training procedure, linearly increasing the learning rate while observing the value of
the loss function. Finally, we employ, as only regularization methodology, "Dropout" [ Srivastava(2014) ] in the time
representation stage, inserted between the LSTM and Time-Distributed layer. This simple tweak allows training a more
robust and resilient to noise temporal patterns extraction stage. Indeed, forcing CNN to work without relying on certain
temporal activations can greatly improve the abstraction of the generated representations distributing the knowledge
across all available units.


**5** **Experimental Results and discussion**


We first processed raw data in order to create a set of **n** pixel samples X = _{_ **X** _i_ _}_ with the related ground truth labels
Y = _{y_ _i_ _}_ . Then, in order to have a visual inspection of the data set, principal component analysis (PCA), one of the most
popular dimensionality reduction algorithms, have been applied to project the training set onto a lower tri-dimensional
hyperplane. Finally, quantitative and qualitative results are discussed with a detail description of the architecture
settings.


**5.1** **Training data**


Sample pixels require to be extracted from the raw data and then reordered in order to feed the devised architecture.
Indeed, the first RNN stage requires data points to be collected in slices of time series. So, we separated labeled pixels
from raw data, and we divided them in chunks of data, forming a tri-dimensional tensor **X** _∈_ R _[i][×][t][×][b]_ for the successive
pre-processing pipeline. In Fig. 8, a visual representation of the data set tensor **X** generation, where fixing the first
dimension **X** _i,_ : _,_ : there are the individual pixel samples _X_ [(] _[i]_ [)] with _t_ = 9 time steps and _b_ = 5 spectral bands. It is worth
to notice that the number of time steps and bands are completely an arbitrary choice dictated by the raw data availability.


Subsequently, we adopted a simple pipeline of two steps to pre-process the data. Stratified sampling has been applied in
order to divide the data set tensor **X**, with shape (92116, 9, 5), in a training and test set. Due to the natural unbalanced
number of instances per class present in the data set Table. 3, this is an important step in order to preserve the same
percentage in the two sets. After selecting a split percentage for the training of 60%, we obtained two tensors **X** _train_
and **X** _test_ with shape (55270, 9, 5) and (36846, 9, 5), respectively. Secondly, as common practice, in order to facilitate
the training, we adopted standard scaling, ( _x −_ _µ_ ) _/σ_, to normalize the two sets of data points.


11


P IXEL R-CNN


_X_ _**[(i)]**_


Figure 8: Overview of the tensor **X** _∈_ R _[i][×][t][×][b]_ generation. The first dimension _i_ represents the collected instances _X_ [(] _[i]_ [)],
the second _t_ the different time steps, and finally the last one _b_ the five spectral bands, red, green, blue, near-infrared and
NDVI. On the top, labeled pixels are extracted simultaneously, from all-time steps and bands starting from the raw
satellite images. Then, _X_ _i,j,k_ are reshaped in order to set up the _X_ [(] _[i]_ [)] = **X** _i,_ : _,_ : of the data set tensor **X** .


Table 3: Land cover types contribution in the reference data.


**Class** **Pixels** **Percentage**


Tomatoes 3020 3.20%
Artificials 9343 10.14%
Trees 7384 8.01%
Rye 4382 4.75%
Wheat 12826 13.92%
Soya 5836 6.33%
Apple 849 0.92%
Peer 495 0.53%
Temp Grass 1744 1.89%
Water 2451 2.66%

Lucerne 17942 19.47%

Drum Wheat 1188 1.28%
Vineyard 6110 6.63%
Barley 2549 2.76%
Maize 15997 17.37%

**Total** 92116 100%


**5.2** **Dataset Visualization**


In order to explore and visualize the generated set of points, we exploit Principle Component Analysis (PCA), reducing
the high dimensionality of the data set. For this operation, we considered the components t and b as features of our
data points. So, applying Singular Value Decomposition (SVD) and then selecting the first three principal components,
_W_ _d_ = ( _c_ 1 _, c_ 2 _, c_ 3 ), it was possible to plot the different classes in a tri-dimensional space, having a visual representation
of the projected data points. In Fig. 9 the projected data points are plotted in tri-dimensional space. Except for water
bodies, it is worth to point out how much intra-class variance is present. Indeed, most of the classes lay on more than
one hyperplane, demonstrating the difficulty of the task and the studied data set. Finally, it was possible to analyze the
explained variance ratio varying the number of dimensions. From Fig. 10, it is worth to notice that approaching higher
components, the explained variance trend stops growing fast. So, that can be considered as the intrinsic dimensionality
of the data set. Due to this fact, it is reasonable to assume that reducing the number of time steps would not significantly
affect the overall results.


12


P IXEL R-CNN


Figure 9: Visual representation of the data points projected in the tri-dimensional space using PCA. The three principal
components took into account preserve 64.5% of the original data set variance.


Figure 10: Pareto Chart of the explained variance as a function of the number of components.


**5.3** **Experimental settings**


In this section, we examine the settings of the final network architecture. The basic Pixel R-CNN model, shown in Fig.
3, is the result of a careful design aimed at obtaining the best performance in terms of accuracy and computational cost.
Indeed, the final model is a lightweight model with 30,936 trainable parameters (less than 1 MB), fast and more accurate
than the existing state-of-the-art solutions. With the suggested approach, we employed only an RNN layer with 32
output units for each peephole LSTM cell randomly turned off, with a probability _p_ = 0.2, by a Dropout regularization
operation. For all experiments, peephole LSTM has shown an improvement in overall accuracy around 0.8% over
standard LSTM cells. Then, Time Distributed Dense transforms _Y_ ( [(] _lstm_ _[i]_ [)] ) [in a 9x9 square matrix that feed a stack of two]
CNN layers with a number of features _n_ 1 = 16 and _n_ 2 = 32, respectively. The first layer as a filter size of _f_ 1 =3 and the
second one _f_ 2 = 7 producing a one-dimensional output array. Finally, a fully connected layer with SoftMax activation
function maps _Y_ ( [(] _conv_ _[i]_ [)] 2) [to the probability of the] _[ K]_ [ = 15] [ different classes. Except for the final layer, we adopted ReLU]
as activation functions. In order to find the best training hyperparameters for the optimizer, we used 10% of the training


13


P IXEL R-CNN


set to perform a random search evaluation, with few epochs, in order to select the most promising parameters. Then,
after this first preliminary phase, the analysis has been focused only on the most promising hyperparameters value,
fine-tuning them with a grid search strategy.


So, for the AMSGrad optimizer we set _β_ 1 = 0 _._ 86, _β_ 2 = 0 _._ 98 and _ϵ_ = 10 _[−]_ [9] . Finally, as previously introduced, with a
preliminary procedure, we linearly increased the learning rate of _η_ while observing the value of the loss function in
order to estimate the initial value of this important hyperparameter. In conclusion, we fed our model with more than
62000 samples for 150 epochs with a batch size of 128 while cyclically varying the learning rate value with a cosine
aneling strategy. All tests have been carried out with the TensorFlow framework on a workstation with 64 GB RAM,
Intel Core i7-9700K CPU, and an Nvidia 2080 Ti GPU.


**5.4** **Classification**


Performance of the classifier was evaluated by user’s accuracy (UA), producer’s accuracy (PA), overall accuracy (OA),
and the kappa coefficient (K) shown in the confusion matrix see Table. 4, which is the most common metric that has
been used for classification tasks [ Kussul(2016), Congalton(1991), Conrad(2014), Skakun(2015) ]. Overall accuracy
indicates the overall performance of our proposed Pixel R-CNN architecture by calculating a ratio between the correctly
classified total number of pixels and total ground truth pixels for all classes. The diagonal elements of the matrix
represent the pixels that were classified correctly for each class. Individual class accuracy was calculated by dividing
the number of correctly classified pixels in each category by the total number of pixels in the corresponding row called
User’s accuracy, and columns called Producer’s accuracy. PA indicates the probability that a certain crop type on the
ground is classified as such. UA represents the probability that a pixel classified in a given class belongs to that class.


Figure 11: (a). Final classified map using Pixel R-CNN, (b). zoomed in region of the classified map, and (c). Raw
Sentinel-2 RGB composite of the zoomed region.


Our proposed pixel-based Pixel R-CNN method achieved OA=96.5% and Kappa=0.914 with 15 number of classes for
a diverse large scale area, which exhibits significant improvement as compared to other mainstream methods. Water
bodies and trees stand highest in terms of UA with 99.1% and 99.3%, respectively. That is mainly due to intra-class
variability and the minor change of NIR band reflectance over the time, which was easily learned by our Pixel R-CNN.
Most of the classes, including the major types of crops such as Maize, Wheat, Lucerne, Vineyard, Soya, Rye, and
Barley, were classified with more than 95% UA. Grassland being the worst class, which was classified with the PA =
65% and UA = 63%. The major confusion of grassland class was with Lucerne and Vineyard. It is worth mentioning
that, Artificial class, which belongs to roads, buildings, urban areas, represents mixed nature of pixel reflectances, was
accurately detected with UA = 97% and PA=99%.


For class Apple, obtained PA was 86% while UA = 68%, which shows that 86% of the ground truth pixels were
identified as Apple, but only 68% of the pixels classified as Apple in the classification were actually belonged to


14


P IXEL R-CNN


Table 4: Obtained confusion matrix.


**Ground Truth** **Classified Classes**
Total PA


TM AR TR RY WH SY AP PR GL WT LN DW VY BL MZ


Tomatoes (TM) **1096** 0 0 0 4 11 0 0 0 0 0 0 0 0 0 1111 98%
Artificial (AR) 0 **3752** 8 1 2 0 2 1 9 9 12 2 6 0 4 3808 99%
Trees (TR) 0 31 **2967** 1 0 0 0 3 10 0 17 0 2 0 0 3031 98%
Rye (RY) 0 1 0 **1960** 25 0 0 0 0 0 0 0 0 5 0 1991 98%
Wheat (WH) 38 7 0 221 **4981** 6 0 0 10 0 14 1 2 38 42 5360 93%
Soya (SY) 3 0 0 0 3 **1226** 0 0 0 0 11 0 3 0 41 1287 95%
Apple (AP) 0 0 0 0 0 0 **142** 0 0 0 2 0 21 0 0 165 86%
Peer (PR) 0 0 11 0 0 0 27 **124** 0 0 0 0 6 0 0 168 73%
Grassland (GL) 0 39 3 7 0 1 0 0 **239** 0 72 0 3 0 4 368 65%
Water (WT) 0 0 0 0 0 0 0 0 0 **906** 0 0 0 0 0 906 100%
Lucerne (LN) 0 0 0 2 0 2 0 0 48 0 **7250** 0 26 0 10 7338 98%
Durum.Wheat (W) 0 4 0 0 0 0 0 0 2 0 0 **322** 0 0 0 328 98%
Vineyard (VY) 11 7 4 4 11 1 50 1 21 0 93 0 **2139** 0 7 2349 91%
Barley (BL) 0 1 0 2 24 0 0 0 1 0 1 0 0 **817** 0 846 96%
Maize (MZ) 17 14 0 0 10 24 0 3 10 0 16 1 6 0 **7689** 7790 99%
Total 1165 3856 2993 2198 5060 1271 221 132 350 915 7488 326 2214 860 7797

UA 94% 97% 99% 89% 98% 96% 64% 93% 68% 99% 96% 99% 96% 95% 98%


Table 5: An overview and performance of recent studies.

**Study** **Details**


**Sensor** **Features** **Classifier** **Accuracy** **Classes**


Our Sentinel-2 BOA Reflectances Pixel R-CNN 96.50% 15
Rußwurm and Körner[ Lawrence(2018) ], Sentinel-2 TOA Reflectances Recurrent Encoders 90% 17
2018

Skakun et al. [Skakun(2015)], 2016 Radarsat-2 + Landsat-8 Optical+SAR NN and MLPs 90% 11
Conrad et al. [Conrad(2014)], 2014 RapidEye Vegetation Indices RF and OBIA 86% 9
Vuolo et al. [Vuolo(2018)], 2018 Sentinel-2 Optical RF 91-95% 9
Hao et al. [Hao(2015)], 2015 MODIS Stat + phenological RF 89% 6
J.M. Pena-Barragán [Peña(2011)], 2011 ASTER Vegetation Indices OBIA+DT 79% 13


class Apple. Some Pixels (see Table. 4 ) belongs to Peer and Vineyard were mistakenly classified as Apple. The
final classified map is shown in Fig. 11 with the example of the zoomed part and actual RGB image. To the best of
our knowledge, a multi-temporal benchmark dataset is not available to compare classification approaches on equal
footings. There are some data sets available online for crop classification without having ground truth of other land
cover types such as Trees, Artificial land (build ups), Water bodies, Grassland. Therefore it is difficult to compare
classification approaches on equal footings. Indeed, a direct quantitative comparison of the classification performed in
these studies is difficult due to various dependencies such as the number of evaluated ground truth samples, the extent
of the considered study area, and the number of classes to be evaluated. Nonetheless, we provided an overview of recent
studies and their performances of the study domain by their applied approaches, the number of considered classes,
used sensors, and achieved overall accuracy in Table. 5. Hao et al. [ Hao(2015) ], achieved 89% OAA by using RF
classifier on the extracted phenological features from MODIS time-series data. They determined that good classification
accuracies can be achieved with handcrafted features and classification algorithms if the temporal resolution of the
data is sufficient. Though, the MODIS sensor data is not suitable for classification of the areas of large homogeneous
regions due to its low spatial resolution (500m). Conrad et al. [ Lawrence(2018) ] used high spatial resolution data from
the RapidEye sensor and achieved 90% OAA for nine considered classes. In [ Conrad(2014) ], features from optical
and SAR were extracted and used by the committee of neural networks of multilayer perceptrons to classify a diverse
agriculture region considerably. Recurrent encoders have been employed in [ Bergstra(2012) ] to classify a large area
for 17 considered classes using high spatial resolution (10m) sentinel-2 data and achieved 90% OAA,which proved
that recurrent encoders are useful to capture the temporal information of spectral features that leads to higher accuracy.
Voulo et al. [ Skakun(2015) ] also used sentinel-2 data and achieved a maximum 95% classification accuracy using RF
classifier but nine classes were considered.


In conclusion, it is interesting to notice neuron activation inside the network during the classification process. Indeed, it
is possible to plot unit values when the network receives specific inputs and compare how model behaviors change. In
Fig. 12 four samples, belonging to the same class "artificials," feed the model creating individual activations in the input
layer. Even if they all belong to the same class, the four instances took into account present a noticeable variance. Either


15


P IXEL R-CNN


(a) (b)


Figure 12: Visual representation of the activation of the internal neurons of Pixel R-CNN, where darker color are values
close to zero and vice versa. (a). four samples of the same class "artificials", (b). related activations inside the network
at the output of the TimeDistributedDense layer _Y_ ( _timeD_ ) . It is interesting to notice how the four inputs are pretty
different from each other, but the network representations already at this level are similar.


the spectral features in a specific time instance (rows) or their temporal variation (columns) present different patterns
that make them difficult to classify. However, already after the first LSTM layer with the TimeDistributedDense block,
the resulting 9x9 matrices _Y_ ( _timeD_ ) have a clear pattern that can be used by the following layers to classify the different
instances in their respective classes. So, the network during the training process learns to identify specific temporal
schemes, that allows making strong distributed and disentangle representations.


**5.5** **Non deep learning classifiers**


We tried four other traditional classifiers on the same dataset for comparison, which are Support Vector Machine (SVM),
Kernal SVM, Random Forest (RF), and XGBoost. These are well-known classifiers for their high performances and
also considered as baseline models in classification tasks [ Srivastava(2014) ]. SVM can perform nonlinear classification
using kernel functions by separating hyperplanes. A widely used RF classifier is an ensemble of decision trees based on
the bagging approach [ Fernández(2014), Shi(2016) ]. XGBoost is state of the art classifier based on gradient boosting
model of decision trees, which attracted much attention in the machine learning community. RF and SVM have
been widely used in remote sensing applications [ Löw(2013), Hao(2015), Gislason(2006) ]. Each classifier involves
hyperparameters that need to be tuned at the time of classification model development.


We followed "random search" approach to optimize major hyperparameters [ Lawrence(2006) ]. Best values of hyperparameters were selected based on classification accuracy achieved for the validation set, and are highlighted with
bold letters in Table. 6. Further details about hyper parameters and achieved overall accuracy (OA) for SVM, Kernal
SVM, RF, and XGBoost are reported in Table. 6 From these non-deep learning classifiers, SVM stands highest with
OA = 79.6% while RF, Kernel SVM, and XGBoost achieved 77.5%, 76.8% and 77.2% respectively. From the results
presented in Table. 6, our proposed Pixel R-CNN based classifier achieved OA = 96.5%, which is far better results than
the non deep learning classifiers. Learning temporal and spectral correlations from multi-temporal images considering
large data set is challenging for traditional non-deep learning techniques. The introduction of deep learning models in
the remote sensing domain brought more flexibility to exploit temporal features in such a way that it can increase the
amount of information to gain much better and reliable results for classification tasks.


**6** **Conclusion**


In this study, we developed a novel deep learning model with Recurrent and Convolutional Neural Network called
Pixel R-CNN to perform Land Cover and Crop Classification by using multitemporal decametric sentinel-2 imagery of
central north part of Italy. Our proposed Pixel R-CNN based architecture exhibits significant improvement as compared
to other mainstream methods by achieving 96.5% overall accuracy with kappa=0.914 for 15 number of classes. We also


16


P IXEL R-CNN



Table 6: Comparison of Pixel R-CNN with non-deep learning classifiers.


**Model** **Parameters** **OA**


SVM C: 0.01, 0.1, **1**, 10, 100, 1000 79.50%
Kernel: linear


Kernel SVM C: 0.01, 0.1, **1**, 10, 100, 1000 76.20%
Kernel: **rbf**
Gamma: **0.1**, 0.2, 0.3, 0.4, 0.5, 0.6,
0.7, 0.8



Random Forest n_estimators: 10, 20, 100, 200,
**500** max_depth: **5**, 10, 15, 30
min_samples_split: 3, **5**, 10, 15, 30
min_samples_leaf: 1, 3, **5**, 10



77.90%



XGBoost learning_rate: **0.01**, 0.02, 0.05, 0.1 77.60%
gamma: 0.05, **0.1**, 0.5, 1
max_depth: 3, **7**, 9, 20, 25
min_child_weight: 1, 5, 7, **9**
subsamples: 0.5, **0.7**, 1
colsample_bytree: **0.5**, 0.7, 1
reg_labda: 0.01, 0.1, **1**
reg_alpha: 0, 0.1, 0.5, **1**


**Pixel R-CNN** Mentioned in experimental settings **96.50%**


tested widely used non-deep learning classifiers such as SVM, RF, SVM kernel, and XGBoost to compare with our
proposed classifier and revealed that these methods are less effective, especially when the temporal features extraction
is the key to increase classification accuracy. The main advantage of our architecture is the capability of automated
features extraction by learning time correlation of multiple images, which reduces manual feature engineering and
modeling crops phenological stages.


**Acknowledgments**


This work has been developed with the contribution of the Politecnico di Torino Interdepartmental Centre for Service
Robotics PIC4SeR (https://pic4ser.polito.it) and SmartData@Polito (https://smartdata.polito.it).


**References**


[Gomez(2016)] Gomez, C., White, J. C., Wulder, M. A. Optical remotely sensed time series data for land cover
classification: A review. _ISPRS J. Photogramm. Remote Sens._ **2016**, _116_, 55-72.


[Wu(2014)] Wu, W.-B.; Yu, Q.-Y.; Peter, V.H.; You, L.-Z.; Yang, P.; Tang, H.-J. How Could Agricultural Land Systems
Contribute to Raise Food Production Under Global Change? _J Integr. Agric._ **2014**, 13, 1432–1442.


[Jin(2019)] Jin, Z.; Azzari, G.; You, C.; Di Tommaso, S.; Aston, S.; Burke, M.; Lobell, D.B. Smallholder maize area
and yield mapping at national scales with Google Earth Engine. _Remote Sens. Environ_ . **2019**, 228, 115–128.


[Yang(2007)] Yang, P.; Wu, W.-B.; Tang, H.-J.; Zhou, Q.-B.; Zou, J.-Q.; Zhang, L. Mapping Spatial and Temporal
Variations of Leaf Area Index for Winter Wheat in North China. _Agric. Sci. China_, **2007**, 6, 1437–1443.


[Wang(2016)] Wang, L.A.; Zhou, X.; Zhu, X.; Dong, Z.; Guo, W. Estimation of biomass in wheat using random forest
regression algorithm and remote sensing data. _Crop. J._ **2016**, 4, 212–219.


[Guan(2016)] Guan, K.; Berry, J.A.; Zhang, Y.; Joiner, J.; Guanter, L.; Badgley, G.; Lobell, D.B. Improving the
monitoring of crop productivity using spaceborne solar-induced fluorescence. _Glob. Chang. Biol._ **2016**, 22,
716–726.


[Matton(2015)] Matton, N., Canto, G., Waldner, F., Valero, S., Morin, D., Inglada, J., Defourny, P. An automated
method for annual cropland mapping along the season for various globally-distributed agrosystems using high
spatial and temporal resolution time series. _Remote Sens. (Basel)_ **2015**, _7(10)_, 13208-13232.


17


P IXEL R-CNN


[Battude(2016)] Battude, M.; Al Bitar, A.; Morin, D.; Cros, J.; Huc, M.; Marais Sicre, C.; Le Dantec, V.; Demarez, V.
Estimating maize biomass and yield over large areas using high spatial and temporal resolution Sentinel-2 like
remote sensing data. _Remote Sens. Environ._ **2016**, 184, 668–681.

[Huang(2015)] Huang, J.; Tian, L.; Liang, S.; Ma, H.; Becker-Reshef, I.; Huang, Y.; Su, W.; Zhang, X.; Zhu, D.; Wu,
W. Improving winter wheat yield estimation by assimilation of the leaf area index from Landsat TM and MODIS
data into the WOFOST model. _Agric. For. Meteorol._ **2015**, 204, 106–121.

[Toureiro(2017)] Toureiro, C.; Serralheiro, R.; Shahidian, S.; Sousa, A. Irrigation management with remote sensing:
Evaluating irrigation requirement for maize under Mediterranean climate condition. _Agric. Water Manag._ **2017**, 184,
211–220.

[Yu(2017)] Yu, Q.; Shi, Y.; Tang, H.; Yang, P.; Xie, A.; Liu, B.; Wu, W. eFarm: A Tool for Better Observing
Agricultural Land Systems. _Sensors_, **2017**, 17, 453.

[Boryan(2011)] Boryan, C., Yang, Z., Mueller, R., Craig, M. Monitoring US agriculture: the US department of
agriculture, national agricultural statistics service, cropland data layer program. _Geocarto. Int._ **2011**, _26(5)_, 341358.

[Xie(2008)] Xie, Y., Sha, Z., Yu, M. _Remote sensing imagery in vegetation mapping: a review. Plant Ecol._ **2008**, _1(1)_,
9-23.

[Johnson(2016)] Johnson, D. M. (2016). A comprehensive assessment of the correlations between field crop yields and
commonly used MODIS products. _Int. J. Appl. Earth Obs. Geoinf._ **2016**, _52_, 65-81.

[Senf(2015)] Senf, C., Leitão, P. J., Pflugmacher, D., van der Linden, S., Hostert, P. Mapping land cover in complex
Mediterranean landscapes using Landsat: Improved classification accuracies from integrating multi-seasonal and
synthetic imagery. _Remote Sens. Environ._ **2015**, _156_, 527-536.

[Jin(2016)] Jin, H., Li, A., Wang, J., Bo, Y. Improvement of spatially and temporally continuous crop leaf area index
by integration of CERES-Maize model and MODIS data. _Eur. J. Agron._ **2016**, _78_, 1-12.

[Liaqat(2017)] Liaqat, M. U., Cheema, M. J. M., Huang, W., Mahmood, T., Zaman, M., Khan, M. M. Evaluation of
MODIS and Landsat multiband vegetation indices used for wheat yield estimation in irrigated Indus Basin. _Comput._
_Electron. Agric._ **2017**, _138_, 39-47.

[Senf(2017)] Senf, C., Pflugmacher, D., Heurich, M., Krueger, T. A Bayesian hierarchical model for estimating
spatial and temporal variation in vegetation phenology from Landsat time series. _Remote Sens. Environ._ **2017**, _194_,
155-160.

[Kussul(2016)] Kussul, N., Lemoine, G., Gallego, F. J., Skakun, S. V., Lavreniuk, M., Shelestov, A. Y. Parcel-based
crop classification in ukraine using landsat-8 data and sentinel-1A data. _IEEE J. Sel. Top. Appl. Earth Obs. Remote_
_Sens._ **2016**, _9(6)_, 2500-2508.

[Xiong(2017)] Xiong, J., Thenkabail, P. S., Gumma, M. K., Teluguntla, P., Poehnelt, J., Congalton, R. G., Thau,
D. Automated cropland mapping of continental Africa using Google Earth Engine cloud computing. _ISPRS J._
_Photogramm. Remote Sens._ **2017**, _126_, 225-244.

[Yan(2015)] Yan, L., Roy, D. P. Improved time series land cover classification by missing-observation-adaptive
nonlinear dimensionality reduction. _Remote Sens. Environ._ **2015**, _158_, 478-491.

[Gomez(2016)] Gomez, C., White, J. C., Wulder, M. A. Optical remotely sensed time series data for land cover
classification: A review. _ISPRS J. Photogramm. Remote Sens._ **2016**, _116_, 55-72.

[Xiao(2018)] Xiao, J., Wu, H., Wang, C., Xia, H. Land Cover Classification Using Features Generated From Annual
Time-Series Landsat Data. _IEEE Geosci. Remote Sens. Lett._ **2018**, _15(5)_, 739-743.

[Khaliq(2018)] Khaliq, A., Peroni, L., Chiaberge, M. Land cover and crop classification using multitemporal Sentinel-2
images based on crops phenological cycle. In _IEEE Workshop on Environmental, Energy, and Structural Monitoring_
_Systems (EESMS)_ IEEE, 2018; pp. 1-5.

[Gallego(2008)] Gallego, J., Craig, M., Michaelsen, J., Bossyns, B., Fritz, S. _Best practices for crop area estimation_
_with remote sensing._ **2008**, Ispra: Joint Research Center.

[Zhou(2013)] Zhou, F., Zhang, A., Townley-Smith, L. A data mining approach for evaluation of optimal time-series of
MODIS data for land cover mapping at a regional level. _ISPRS J. Photogramm. Remote Sens._ **2013**, _84_, 114-129.

[Arvor(2011)] Arvor, D., Jonathan, M., Meirelles, M. S. P., Dubreuil, V., Durieux, L. Classification of MODIS EVI
time series for crop mapping in the state of Mato Grosso, Brazil. _Remote Sens._ **2011**, _32(22)_, 7847-7871.

[Zhong(2012)] Zhong, L., Gong, P., Biging, G. S. Phenology-based crop classification algorithm and its implications
on agricultural water use assessments in California’s Central Valley. _Photogramm. Eng. Remote Sens._ **2012**, _78(8)_,
799-813.


18


P IXEL R-CNN


[Wardlow(2012)] Wardlow, B. D., Egbert, S. L. A comparison of MODIS 250-m EVI and NDVI data for crop mapping:
a case study for southwest Kansas. _Int. J. Remote Sens._ **2012**, _31(3)_, 805-830.


[Löw(2013)] Löw, F., Michel, U., Dech, S., Conrad, C. Impact of feature selection on the accuracy and spatial
uncertainty of per-field crop classification using support vector machines. _ISPRS J. Photogramm. Remote Sens._
**2013**, _85_, 102-119.


[Hao(2015)] Hao, P., Zhan, Y., Wang, L., Niu, Z., Shakir, M. Feature selection of time series MODIS data for early
crop classification using random forest: A case study in Kansas, USA. _Remote Sens._ **2015**, _7(5)_, 5347-5369.


[Blaschke(2010)] Blaschke, T. Object based image analysis for remote sensing. _ISPRS J. Photogramm. Remote Sens._
**2010**, _65(1)_, 2-16.


[Novelli(2016)] Novelli, A., Aguilar, M. A., Nemmaoui, A., Aguilar, F. J., Tarantino, E. Performance evaluation of
object based greenhouse detection from Sentinel-2 MSI and Landsat 8 OLI data: A case study from Almería
(Spain). _Int. J. Appl. Earth Obs. Geoinf._ **2016,** _52_, 403-411.


[Long(2013)] Long, J. A., Lawrence, R. L., Greenwood, M. C., Marshall, L., Miller, P. R. Object-oriented crop
classification using multitemporal ETM+ SLC-off imagery and random forest. _GIsci. Remote Sens._ **2013**, _50(4)_,
418-436.


[Li(2015)] Li, Q., Wang, C., Zhang, B., Lu, L. (2015). Object-based crop classification with Landsat-MODIS enhanced
time-series data. _Remote Sens. (Basel)_ **2015**, _7(12)_, 16091-16107.


[Walker(2014)] Walker, J. J., De Beurs, K. M., Wynne, R. H. Dryland vegetation phenology across an elevation
gradient in Arizona, USA, investigated with fused MODIS and Landsat data. _Remote Sens. Environ._ **2014**, _144_,
85-97.


[Walker(2015)] Walker, J. J., De Beurs, K. M., Henebry, G. M. Land surface phenology along urban to rural gradients
in the US Great Plains. _Remote Sens. Environ._ **2015**, _165_, 42-52.


[Simonneaux(2008)] Simonneaux, V., Duchemin, B., Helson, D., Er-Raki, S., Olioso, A., Chehbouni, A. G. The use of
high-resolution image time series for crop classification and evapotranspiration estimate over an irrigated area in
central Morocco. _Int. J. Remote Sens._ **2008**, _29(1)_, 95-116.


[Shao(2016)] Shao, Y., Lunetta, R. S., Wheeler, B., Iiames, J. S., Campbell, J. B. An evaluation of time-series smoothing
algorithms for land-cover classifications using MODIS-NDVI multi-temporal data. _Remote Sens. Environ._ **2016**,
_174_, 258-265.


[Galford(2008)] Galford, G. L., Mustard, J. F., Melillo, J., Gendrin, A., Cerri, C. C., Cerri, C. E. Wavelet analysis of
MODIS time series to detect expansion and intensification of row-crop agriculture in Brazil. _Remote Sens. Environ._
**2008**, _112(2)_, 576-587.


[Funk(2009)] Funk, C., Budde, M. E. Phenologically-tuned MODIS NDVI-based production anomaly estimates for
Zimbabwe. _Remote Sens. Environ._ **2009**, _113(1)_, 115-125.


[Siachalou(2015)] Siachalou, S., Mallinis, G., Tsakiri-Strati, M. A hidden Markov models approach for crop classification: Linking crop phenology to time series of multi-sensor remote sensing data. _Remote Sens. (Basel)_ **2015**, _7(4)_,
3633-3650. _Remote Sens. Environ._ **2009**, _113(1)_, 115-125.


[Xin(2015)] Xin, Q., Broich, M., Zhu, P., Gong, P. Modeling grassland spring onset across the Western United States
using climate variables and MODIS-derived phenology metrics. _Remote. Sens. Environ._ **2015**, _161_, 63-77.


[Xin(2016)] Gonsamo, A., Chen, J. M. Circumpolar vegetation dynamics product for global change study. _Remote._
_Sens. Environ._ **2016**, _182_, 13-26.


[Dannenberg(2015)] Dannenberg, M. P., Song, C., Hwang, T., Wise, E. K. Empirical evidence of El Niño–Southern
Oscillation influence on land surface phenology and productivity in the western United States. _Remote Sens. Environ._
**2015**, _159_, 167-180.


[Khatami(2016)] Khatami, R., Mountrakis, G., Stehman, S. V. A meta-analysis of remote sensing research on supervised pixel-based land-cover image classification processes: General guidelines for practitioners and future research.
_Remote Sens. Environ._ **2016**, _177_, 89-100.


[Gislason(2006)] Gislason, P. O., Benediktsson, J. A., Sveinsson, J. R. Random forests for land cover classification.
_Pattern Recognit. Lett._ **2006**, _27(4)_, 294-300.


[LeCun(2015)] LeCun, Y., Bengio, Y., Hinton, G. Deep learning. _Nature_ **2015**, _521(7553)_, 436-444.


[Congalton(1991)] Congalton, R. G. A review of assessing the accuracy of classifications of remotely sensed data.
_Remote Sens. Environ._ **1991**, _37(1)_, 35-46.


19


P IXEL R-CNN


[Sutskever(2014)] Sutskever, I., Vinyals, O., Le, Q. V. Sequence to sequence learning with neural networks. In _Adv._
_Neural Inf. Process Syst._ ; Nips, 2014; pp. 3104-3112.

[Sutskever(2000)] Gers, F. A., Schmidhuber, J., Cummins, F. Learning to forget: Continual prediction with LSTM.
_Neural Computation_ **2000**, 850-855

[Chung(2014)] Chung, J., Gulcehre, C., Cho, K., Bengio, Y. Empirical evaluation of gated recurrent neural networks
on sequence modeling. _arXiv preprint arXiv_ **2014**, 1412.3555.

[Bahdanau(2014)] Bahdanau, D., Cho, K., Bengio, Y.. Neural machine translation by jointly learning to align and
translate. _arXiv preprint arXiv_ **2014**,1409.0473.

[Chung(2015)] Chung, J., Gulcehre, C., Cho, K., Bengio, Y Gated feedback recurrent neural networks. In _Proc. Int._
_Conf. Machine Learning_, 2015, pp. 2067-2075.

[Lyu(2016)] Lyu, H., Lu, H., Mou, L. Learning a transferable change rule from a recurrent neural network for land
cover change detection. _Remote Sens._ **2016**, _8(6)_, 506.

[Lyu(2018)] Lyu, H., Lu, H., Mou, L., Li, W., Wright, J., Li, X., Gong, P. Long-term annual mapping of four cities on
different continents by applying a deep information learning method to landsat data. _Remote Sens._ **2018**, _10(3)_, 471.

[LeCun(1998)] LeCun, Y., Bottou, L., Bengio, Y., Haffner, P. Gradient-based learning applied to document recognition.
_Proceedings of the IEEE_ **1998**, _86(11)_, 2278-2324.

[Hubel(1962)] Hubel, D. H., Wiesel, T. N. Receptive fields, binocular interaction and functional architecture in the
cat’s visual cortex. _J. Physiol. (Lond.)_ **1962**, _160(1)_, 106-154.

[Hubel(2010)] Szeliski, R. Computer vision: algorithms and applications. In _Springer Science & Business Media_ ;
Springer, London, 2010.

[Hubel(2015)] Dalal, N., Triggs, B. Histograms of oriented gradients for human detection. _Conf. on Comp. Vision and_
_Patt. Recog_ 2015, 886-893

[Krizhevsky(2012)] Krizhevsky, A., Sutskever, I., Hinton, G. E. Imagenet classification with deep convolutional neural
networks. In _Adv. Neural Inf. Process Syst._ ; 2012, pp. 1097-1105.

[Zeiler(2014)] Zeiler, M. D., Fergus, R. Visualizing and understanding convolutional networks. In _Comput. Vis. ECCV_ ;
Springer, Cham, 2014, pp. 818-833.

[Wan(2017)] Wan, X., Zhao, C., Wang, Y., Liu, W. (2017). Stacked sparse autoencoder in hyperspectral data classification using spectral-spatial, higher order statistics and multifractal spectrum features. _Infrared Phys. & Technol._
**2017**, _86_, 77-89.

[Mou(2017)] Mou, L., Ghamisi, P., Zhu, X. X. (2017). Unsupervised spectral–spatial feature learning via deep residual
Conv–Deconv network for hyperspectral image classification. _IEEE Trans. Geosci. Remote Sens._ **2017**, _56(1)_,
391-406.

[Kampffmeyer(2016)] Kampffmeyer, M., Salberg, A. B., Jenssen, R. Semantic segmentation of small objects and
modeling of uncertainty in urban remote sensing images using deep convolutional neural networks. In _Proceedings_
_of the IEEE conf. on comput. vision and patt. recog. workshops_ ; 2016, pp. 1-9.

[Kampffmeyer(2017)] Maggiori, E., Tarabalka, Y., Charpiat, G., Alliez, P. High-resolution aerial image labeling with
convolutional neural networks. _IEEE Trans. Geosci. Remote Sens._ **2017**, _55(12)_, 7092-7103.

[Audebert(2018)] Audebert, N., Le Saux, B., Lefèvre, S. Beyond RGB: Very high resolution urban remote sensing
with multimodal deep networks. I _ISPRS J Photogramm. Remote Sens._ **2018**, _140_, 20-32.

[Kussul(2017)] Kussul, N., Lavreniuk, M., Skakun, S., Shelestov, A. Deep learning classification of land cover and
crop types using remote sensing data. _IEEE Geosci. Remote Sens. Lett._ **2017**, _14(5)_, 778-782.

[Kussul(2017)] Land Use and Coverage Area frame Survey (LUCAS) Details. [online] Availble: `[https://ec.europa.](https://ec.europa.eu/eurostat/web/lucas)`
`[eu/eurostat/web/lucas](https://ec.europa.eu/eurostat/web/lucas)` [Accessed on 14-01-2019]

[Kussul(2017)] Sentinel-2 MSI Technical Guide: [online] Availble: `[https://sentinel.esa.int/web/sentinel/](https://sentinel.esa.int/web/sentinel/technical-guides/sentinel-2-msi)`
`[technical-guides/sentinel-2-msi](https://sentinel.esa.int/web/sentinel/technical-guides/sentinel-2-msi)` [Accessed on 14-01-2019]

[Kaufman(1988)] Kaufman, Y. J., Sendra, C. Algorithm for automatic atmospheric corrections to visible and near-IR
satellite imagery. _Int. J. Remote Sens._ **1988**, _9(8)_, 1357-1381.

[Mellet(1996)] Mellet, E., Tzourio, N., Crivello, F., Joliot, M., Denis, M., Mazoyer, B. Functional anatomy of spatial
mental imagery generated from verbal instructions. _Open J. Neurosci._ **1996**, _16(20)_, 6504-6512.

[Reddi(2019)] Reddi, S. J., Kale, S., Kumar, S. On the convergence of adam and beyond. _arXiv preprint arXiv_ **2019**,
1904.09237.


20


P IXEL R-CNN


[Kingma(2019)] D.P. Kingma, and J. Ba, "Adam: A method for stochastic optimization", 2014, _arXiv preprint_
_[arXiv:1412.6980.](http://arxiv.org/abs/1412.6980)_ [online] Availble: `[https://arxiv.org/pdf/1412.6980.pdf](https://arxiv.org/pdf/1412.6980.pdf)` [Accessed on [21-04-2019]

[Smith(2017)] Smith, L. N. Cyclical learning rates for training neural networks. In _2017 IEEE Winter Conf. on Appl. of_
_Comput. Vision (WACV)_ ; IEEE, 2017, pp. 464-472.

[Srivastava(2014)] Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., Salakhutdinov, R. Dropout: a simple way
to prevent neural networks from overfitting. _J. Mach. Learn. Res._ **2014**, _15(1)_, 1929-1958.

[Fernández(2014)] Fernández-Delgado, M., Cernadas, E., Barro, S., Amorim, D. Do we need hundreds of classifiers to
solve real world classification problems?. _J. Mach. Learn. Res._ **2014**, _15(1)_, 3133-3181.

[Shi(2016)] Shi, D., Yang, X. An assessment of algorithmic parameters affecting image classification accuracy by
random forests. _Photogramm. Eng. Remote Sens._ **2016**, _82(6)_, 407-417.

[Lawrence(2006)] Lawrence, R. L., Wood, S. D., Sheley, R. L. Mapping invasive plants using hyperspectral imagery
and Breiman Cutler classifications (RandomForest). _Remote Sens. Environ._ **2006**, _100(3)_, 356-362.

[Bergstra(2012)] Bergstra, J., Bengio, Y. Random search for hyper-parameter optimization. _J. Mach. Learn. Res._ **2012**,
_13_, 281-305.

[Lawrence(2018)] Rußwurm, M., Körner, M. Multi-temporal land cover classification with sequential recurrent
encoders. _ISPRS Int. J. Geoinf._ **2018**, _7(4)_, 129.

[Conrad(2014)] Conrad, C., Dech, S., Dubovyk, O., Fritsch, S., Klein, D., Löw, F., Zeidler, J. Derivation of temporal
windows for accurate crop discrimination in heterogeneous croplands of Uzbekistan using multitemporal RapidEye
images. _Comput. Electron. Agric._ **2014**, _103_, 63-74.

[Skakun(2015)] Skakun, S., Kussul, N., Shelestov, A. Y., Lavreniuk, M., Kussul, O. Efficiency assessment of multitemporal C-band Radarsat-2 intensity and Landsat-8 surface reflectance satellite imagery for crop classification in
Ukraine. _IEEE J. Sel. Top. Appl. Earth Obs. Remote Sens._ **2015**, _9(8)_, 3712-3719.

[Vuolo(2018)] Vuolo, F., Neuwirth, M., Immitzer, M., Atzberger, C., Ng, W. T. How much does multi-temporal
Sentinel-2 data improve crop type classification?. _Int. J. Appl. Earth Obs. Geoinf._ **2018**, _72_, 122-130.

[Peña(2011)] Peña-Barragán, J. M., Ngugi, M. K., Plant, R. E., Six, J. Object-based crop identification using multiple
vegetation indices, textural features and crop phenology. _Remote sens. Environ._ **2011**, _115(6)_, 1301-1316.


21


