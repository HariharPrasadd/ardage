# **Multimodal Machine Learning-based Knee** **Osteoarthritis Progression Prediction from Plain** **Radiographs and Clinical Data**

**Aleksei Tiulpin** [1,8,*] **, Stefan Klein** [2] **, Sita M.A. Bierma-Zeinstra** [3,4] **, J´erˆome Thevenot** [1] **, Esa**
**Rahtu** [5] **, Joyce van Meurs** [6] **, Edwin H.G. Oei** [7] **, and Simo Saarakkala** [1,8]


1 Research Unit of Medical Imaging, Physics and Technology, University of Oulu, Oulu, Finland.
2 Biomedical Imaging Group Rotterdam, Depts. of Medical Informatics & Radiology, Erasmus MC, University
Medical Center Rotterdam, the Netherlands.
3 Department of General Practice, Erasmus MC, University Medical Center Rotterdam, the Netherlands
4 Department of Orthopedics, Erasmus MC, University Medical Center Rotterdam, the Netherlands.
5 Department of Signal Processing, Tampere University of Technology, Tampere, Finland.
6 Department of Internal Medicine, Erasmus MC, University Medical Center Rotterdam, the Netherlands
7 Department of Radiology & Nuclear Medicine, University Medical Center Rotterdam, the Netherlands
8 Department of Diagnostic Radiology, Oulu University Hospital, Oulu, Finland
- aleksei.tiulpin@oulu.fi


**ABSTRACT**


Knee osteoarthritis (OA) is the most common musculoskeletal disease without a cure, and current treatment options are
limited to symptomatic relief. Prediction of OA progression is a very challenging and timely issue, and it could, if resolved,
accelerate the disease modifying drug development and ultimately help to prevent millions of total joint replacement surgeries
performed annually. Here, we present a multi-modal machine learning-based OA progression prediction model that utilizes raw
radiographic data, clinical examination results and previous medical history of the patient. We validated this approach on an
independent test set of 3,918 knee images from 2,129 subjects. Our method yielded area under the ROC curve (AUC) of 0.79
(0.78-0.81) and Average Precision (AP) of 0.68 (0.66-0.70). In contrast, a reference approach, based on logistic regression,
yielded AUC of 0.75 (0.74-0.77) and AP of 0.62 (0.60-0.64). The proposed method could significantly improve the subject
selection process for OA drug-development trials and help the development of personalized therapeutic plans.


**Introduction**


Knee osteoarthritis (OA) is the most common musculoskeletal disorder causing significant disability for patients worldwide [1] .
OA is a degenerative disease and there is a lack of knowledge on the factors contributing to its progression. The overall etiology
of OA is also not understood and there is no effective treatment, besides behavioral interventions. Furthermore, at the end stage
of the disease, the only available treatment option is total knee replacement (TKR) surgery, which is highly invasive, costly and
also strongly affects the patient’s quality of life. OA is a major burden for the public health care system and it is increasing
further with the aging of the population. For example, according to the statistics only in the United States, around 12% of the
population suffer from OA and the annual rate of TKR for people 45-64 years of age has doubled since the year of 2000 [2] . From
the economical point of view, OA causes enormous costs for society and the costs of these surgeries are estimated to be over
nine billion euros [2] .

In primary health care, OA is currently diagnosed based on a combination of clinical history, physical examination, and
X-ray imaging (radiography) if needed. However, the current widely available diagnostic modalities do not allow for effective
OA prognosis assessment [3], which is important for the planning of appropriate therapeutic interventions and also for recruitment
to OA disease modifying drugs development trials [4] . A possible improvement would be to extend this diagnostic chain with
Magnetic Resonance Imaging (MRI), which is, however, costly, time-consuming, has limited availability and not applicable for
wide use [5] .

While being imperfect and lacking decision consistency, the current OA diagnostic tools can be enhanced using computerassisted methods. For example, it has been shown that the gold clinical standard for OA severity assessment from radiographs,
semi-quantitative Kellgren-Lawrence (KL) [6] system that highly suffers from subjectivity of a practitioner, can be automated
using Deep Learning – a state-of-the-art Machine Learning approach widely used in computer vision [7][–][9] . However, to the best


of our knowledge, there have been no similar studies on Deep Learning-based prediction of structural knee OA progression, in
which the raw image data are directly used for prediction instead of the KL grades defined by a radiologist.
Current state-of-the-art OA progression prediction models are based on a combination of texture descriptors that are
calculated from imaging, KL-grade, clinical and anthropometric data [10][–][13] . However, their performance and generalizability are
difficult to assess for multiple reasons. Firstly, the texture descriptors may suffer from sensitivity to data acquisition settings.
This can lead to limited sample size, as is for example seen in the studies of Janvier _et al._, where only non-processed digital
images were used [11][,] [12] . Secondly, only a few current progression studies used an external dataset besides the one that was
utilized to develop the prediction model [10][,] [14][,] [15] . If such external dataset is not utilized, this can lead to a possible overfitting and
eventually a bias in the final results [7] . Finally, it has been previously shown that most of the OA evolution modelling studies
tend to focus on estimating the decrease of joint space width (JSW) as a measure of progression [16] . Such outcome can be
challenging to validate due to inherent problems associated with radiographic data acquisition (e.g. varying beam angle) and it
does not depict all the changes happening with the joint. Previously, it has been recommended to assess OA progression using
measures that incorporate the information about both – JSW and osteophytes [17], _i.e._, treating future increase of the KL-grade as
a progression outcome.
In this study, we propose a novel method based on Machine Learning that directly utilizes raw radiographic data, physical
examination, patient’s medical history, anthropometric data and, optionally, a radiologist’s statement (KL-grade) to predict
structural OA progression. Here, we aim to predict any increase of a current KL-grade or potential need for TKR within
the next 7 years after the baseline examination for patients having no, early or moderate OA. Our method employs a Deep
Convolutional Neural Network (CNN) [18][,] [19] that evaluates the probability of OA progression jointly with the current OA severity
in the analyzed knee as an auxiliary outcome. Further, we improve the prognosis from CNN by fusing its prediction with the
clinical data using a Gradient Boosting Machine (GBM) [20] . Schematically, our method is presented in Figure 1.


**Results**


**Training and testing datasets**
We used the metadata provided in Osteoarthritis Initiative (OAI) and Multicenter Osteoarthritis Study (MOST) cohorts to select
progressors and non-progressors for train and test datasets, respectively. We considered only the knees having no, early or
moderate OA (KL-0, KL-1, KL-2 and KL-3) at the baseline (first visit) as these are the most relevant clinical cases. Furthermore,
we excluded from the test set all the subjects who died between the follow-ups for coherence of our data. Additionally, the
subjects who did not progress and dropped out from the study before the last follow-up examination were excluded. After the
pre-selection process, we used 4,928 knees (2,711 subjects) from OAI dataset for training and 3,918 knees (2,129 subjects)
from MOST dataset for testing of our model. Here, 1,331 (27%) and 1,501 (47%) knees were identified as progressors in OAI
and MOST data, respectively. As a progression definition, we utilized an increase of a KL-grade within the following years.
Here, we ignored the increase from KL-0 to KL-1 and included all cases with progression to TKR. To harmonize the data
between OAI and MOST datasets, we defined the following three fine-grained categories:


_• y_ = 0: no knee OA progression


_• y_ = 1: progression within the next 60 months (fast progression)


_• y_ = 2: progression after 60 months (slow progression)


Supplementary Tables 1 and 2 describe the training and the testing sets derived from OAI and MOST datasets respectively.


**Reference methods**

Firstly, we utilized several reference methods (see details in Methods) in order to understand the added value of our approach.
These models were trained to predict a probability _P_ ( _y >_ 0 _|x_ ) of a particular knee _x_ to have a KL-grade increase in the future.
Here, we pooled the classes _y_ = 1 and _y_ = 2 together to derive a binary outcome, which was used in both Logistic Regression
(LR) and GBM reference methods. In Figure 2, we demonstrate the performance of LR, which is commonly used in OA
research [10][,] [11][,] [14][,] [15] . All of the LR models were derived and tested on the existing image assessment and clinical data provided
by the OAI and MOST datasets, respectively. In cross-validation experiments on OAI data, we also assessed the added value of
regularization [21] and found no difference between regularized and non-regularized LR models.
From Figure 2, it can be seen that two best models exist: one based on Age, Sex, Body-Mass Index and KL grade (model
1), and the other being the same with the addition of symptomatic assessment (Western Ontario and McMaster Universities
Arthritis Index, WOMAC [22] ), injury and surgery history (model 2). We chose the latter in our further comparisons because it
performs with higher precision at lower recall while yielding similar performance at other recall levels. This model yielded
AUC of 0.75 (0.74-0.77) and Average Precision (AP) of 0.62 (0.60-0.64). All the mentioned risk factors included into the
reference models were selected on the basis of their use in the previous studies [10][,] [14][,] [15] .


**2/20**


Attention Map

(GradCAM)



















**Figure 1.** Schematic representation of our multi-modal pipeline, predicting the risk of osteoarthritis (OA) progression for a
particular knee. We first use a Deep Convolutional Neural Network (CNN), trained in a multi-task setting to predict the
probability of OA progression (no progression, rapid progression, slow progression) and the current stage of OA defined
according to the Kellgren-Lawrence (KL) scale. Subsequently, we fuse these predictions with patient’s Age, Sex, Body-Mass
Index, given knee injury and surgery history, symptomatic assessment results and, optionally, a KL grade given by a radiologist
using a Gradient Boosting Machine Classifier. After obtaining prediction from CNN, we utilize GradCAM attention maps to
make our method more transparent and highlight the zones in the input knee radiograph, which were considered most important
by the network.


It was hypothesized that LR might not be able to exploit the full potential of the input data (clinical variables and image
assessments), as with this type of model, non-linear relationships within the data cannot be evaluated. Therefore, we utilized a
GBM and trained it to predict the probability of OA progression. Figure 3 demonstrates the performance of models identical to
model 1 and model 2, but trained using GBM instead of LR (model 3 and model 4). Model 4 performed best and obtained the
AUC of 0.76 (0.75-0.78) and AP of 0.63 (0.61-0.65). The full comparisons of the models built using LR and GBM approaches
are summarized in Table 1 and also in Figures 2 and 3.


**Predicting progression from raw image data**
After testing the reference models, we developed a CNN, which allows to directly leverage raw knee DICOM images in an
automatic manner. In contrast to the previous studies, this model was trained in a multi-task setting to predict OA progression
in the index knee and also its current KL-grade from the corresponding X-ray image. In particular, our model consists of a
feature extractor – a pre-trained se-resnext50-32xd model [23] – and two branches, each of which is a fully connected layer (FC),
predicting its own task. One branch of the model predicts a progression outcome and the other branch a KL grade (Figure 1).
In our experiments, we found that prediction of the previously defined fine-grained classes – no ( _y_ = 0 ), fast ( _y_ = 1 ) and
slow ( _y_ = 2 ) progression, while being inaccurate individually, helps to regularize the training of the CNN and leads to better
performance in predicting overall probability of progression _P_ ( _y >_ 0 _|x_ ) within the following years. Having predicted such
binary outcome, our CNN model (model 5) trained using the baseline knee image yielded AUC of 0.76 and AP of 0.56 in a


**3/20**


|1.0|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>~~Age, SEX, BMI, KL, Surg, Inj, WOMAC (0.75 [0.74, 0.77])~~<br>Age, SEX, BMI, KL (0.75 [0.74, 0.77])<br>Age, SEX, BMI, Surg, Inj, WOMAC (0.68 [0.66, 0.69])<br>Age, SEX, BMI (0.65 [0.63, 0.67])||||||
|0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>~~Age, SEX, BMI, KL, Surg, Inj, WOMAC (0.75 [0.74, 0.77])~~<br>Age, SEX, BMI, KL (0.75 [0.74, 0.77])<br>Age, SEX, BMI, Surg, Inj, WOMAC (0.68 [0.66, 0.69])<br>Age, SEX, BMI (0.65 [0.63, 0.67])||||||
|0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>~~Age, SEX, BMI, KL, Surg, Inj, WOMAC (0.75 [0.74, 0.77])~~<br>Age, SEX, BMI, KL (0.75 [0.74, 0.77])<br>Age, SEX, BMI, Surg, Inj, WOMAC (0.68 [0.66, 0.69])<br>Age, SEX, BMI (0.65 [0.63, 0.67])||||||
|0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>~~Age, SEX, BMI, KL, Surg, Inj, WOMAC (0.75 [0.74, 0.77])~~<br>Age, SEX, BMI, KL (0.75 [0.74, 0.77])<br>Age, SEX, BMI, Surg, Inj, WOMAC (0.68 [0.66, 0.69])<br>Age, SEX, BMI (0.65 [0.63, 0.67])||||||
|0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>~~Age, SEX, BMI, KL, Surg, Inj, WOMAC (0.75 [0.74, 0.77])~~<br>Age, SEX, BMI, KL (0.75 [0.74, 0.77])<br>Age, SEX, BMI, Surg, Inj, WOMAC (0.68 [0.66, 0.69])<br>Age, SEX, BMI (0.65 [0.63, 0.67])|~~Age~~<br>Age,<br>Age,<br>Age,|~~SEX, BMI, K~~<br> SEX, BMI, KL<br> SEX, BMI, Su<br> SEX, BMI (0.|~~, Surg, Inj, W~~<br> (0.75 [0.74, <br>rg, Inj, WOMA<br>65 [0.63, 0.67|~~MAC (0.75 ~~<br>0.77])<br>C (0.68 [0.6<br>])|~~0.74, 0.77])~~<br>6, 0.69])|
|0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>~~Age, SEX, BMI, KL, Surg, Inj, WOMAC (0.75 [0.74, 0.77])~~<br>Age, SEX, BMI, KL (0.75 [0.74, 0.77])<br>Age, SEX, BMI, Surg, Inj, WOMAC (0.68 [0.66, 0.69])<br>Age, SEX, BMI (0.65 [0.63, 0.67])||||||


**(a)**



|Age, SEX, BMI, KL (0.61 [0.59, 0.63])<br>Age, SEX, BMI, Surg, Inj, WOMAC (0.56 [0.53, 0.58])<br>Age, SEX, BMI (0.53 [0.51, 0.55])<br>0.8<br>0.6<br>0.4<br>0.2<br>0.0<br>0.0 0.2 0.4 0.6 0.8 1|Age, SEX, BMI, KL<br>Age, SEX, BMI, Su|(0.61 [0.59,<br>rg, Inj, WOM|0.63])<br>AC (0.56 [0.5|3, 0.58])|
|---|---|---|---|---|
|0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br><br>Age, SEX, BMI, KL (0.61 [0.59, 0.63])<br>Age, SEX, BMI, Surg, Inj, WOMAC (0.56 [0.53, 0.58])<br>Age, SEX, BMI (0.53 [0.51, 0.55])|Age, SEX, BMI (0.|53 [0.51, 0.5|])||
|0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br><br>Age, SEX, BMI, KL (0.61 [0.59, 0.63])<br>Age, SEX, BMI, Surg, Inj, WOMAC (0.56 [0.53, 0.58])<br>Age, SEX, BMI (0.53 [0.51, 0.55])|||||
|0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br><br>Age, SEX, BMI, KL (0.61 [0.59, 0.63])<br>Age, SEX, BMI, Surg, Inj, WOMAC (0.56 [0.53, 0.58])<br>Age, SEX, BMI (0.53 [0.51, 0.55])|||||
|0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br><br>Age, SEX, BMI, KL (0.61 [0.59, 0.63])<br>Age, SEX, BMI, Surg, Inj, WOMAC (0.56 [0.53, 0.58])<br>Age, SEX, BMI (0.53 [0.51, 0.55])|||||


**(b)**



**Figure 2.** Assessment of Logistic Regression-based models’ performance. The subplot (a) demonstrates the ROC curves and
the subplot (b) precision-recall curves. Black dashed lines indicate the performance of a random classifier in case of AUC, and
performance of the prediction model based on the dataset labels distribution. The subplots’ legends reflect the benchmarked
models and the values of corresponding metrics with 95% confidence intervals. Here, Area under the ROC curve metric is used
in subplot (a) and Average Precision in subplot (b).


**Table 1.** Summary of the reference models’ performances on the test set. Top performing models are underlined. 95%
confidence intervals are reported in parentheses.


**AUC** **AP**
**Model**


**LR** **GBM** **LR** **GBM**


Age, Sex, BMI 0.65 (0.63-0.67) 0.64 (0.63-0.66) 0.53 (0.51-0.55) 0.52 (0.49-0.54)


Age, Sex, BMI, Injury,
0.68 (0.66-0.69) 0.68 (0.66-0.69) 0.56 (0.53-0.58) 0.56 (0.53-0.58)
Surgery, WOMAC


                                          -                                          KL-grade 0.73 (0.71-0.75) 0.57 (0.55-0.58)


Age, Sex, BMI,
0.75 (0.74-0.77) 0.76 (0.74-0.77) 0.61 (0.59-0.63) 0.61 (0.59-0.63)
KL-grade


Age, Sex, BMI, Injury,       -       0.75 (0.74, 0.77) 0.76 (0.75 0.78) 0.62 (0.60-0.64) 0.63 (0.61 0.65)
Surgery, WOMAC, KL-grade


BMI – Body-Mass Index
WOMAC – Western Ontario and McMaster Universities Arthritis Index

KL-grade – Kellgren-Lawrence grade
AUC – Area Under the Receiver Operating Characteristic Curve
AP – Average Precision
LR – Logistic Regression
GBM – Gradient Boosting Machine


**4/20**


|1.0|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>~~Age, SEX, BMI, KL, Surg, Inj, WOMAC (0.76 [0.75, 0.78])~~<br>Age, SEX, BMI, KL (0.76 [0.74, 0.77])<br>Age, SEX, BMI, Surg, Inj, WOMAC (0.68 [0.66, 0.69])<br>Age, SEX, BMI (0.64 [0.63, 0.66])||||||
|0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>~~Age, SEX, BMI, KL, Surg, Inj, WOMAC (0.76 [0.75, 0.78])~~<br>Age, SEX, BMI, KL (0.76 [0.74, 0.77])<br>Age, SEX, BMI, Surg, Inj, WOMAC (0.68 [0.66, 0.69])<br>Age, SEX, BMI (0.64 [0.63, 0.66])||||||
|0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>~~Age, SEX, BMI, KL, Surg, Inj, WOMAC (0.76 [0.75, 0.78])~~<br>Age, SEX, BMI, KL (0.76 [0.74, 0.77])<br>Age, SEX, BMI, Surg, Inj, WOMAC (0.68 [0.66, 0.69])<br>Age, SEX, BMI (0.64 [0.63, 0.66])||||||
|0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>~~Age, SEX, BMI, KL, Surg, Inj, WOMAC (0.76 [0.75, 0.78])~~<br>Age, SEX, BMI, KL (0.76 [0.74, 0.77])<br>Age, SEX, BMI, Surg, Inj, WOMAC (0.68 [0.66, 0.69])<br>Age, SEX, BMI (0.64 [0.63, 0.66])||||||
|0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>~~Age, SEX, BMI, KL, Surg, Inj, WOMAC (0.76 [0.75, 0.78])~~<br>Age, SEX, BMI, KL (0.76 [0.74, 0.77])<br>Age, SEX, BMI, Surg, Inj, WOMAC (0.68 [0.66, 0.69])<br>Age, SEX, BMI (0.64 [0.63, 0.66])|~~Age, ~~<br>Age, <br>Age, <br>Age,|~~EX, BMI, K~~<br>SEX, BMI, KL<br>SEX, BMI, Su<br>SEX, BMI (0.|~~, Surg, Inj, W~~<br> (0.76 [0.74, <br>rg, Inj, WOMA<br>64 [0.63, 0.66|~~MAC (0.76 ~~<br>0.77])<br>C (0.68 [0.6<br>])|~~0.75, 0.78])~~<br>6, 0.69])|
|0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>~~Age, SEX, BMI, KL, Surg, Inj, WOMAC (0.76 [0.75, 0.78])~~<br>Age, SEX, BMI, KL (0.76 [0.74, 0.77])<br>Age, SEX, BMI, Surg, Inj, WOMAC (0.68 [0.66, 0.69])<br>Age, SEX, BMI (0.64 [0.63, 0.66])||||||


**(a)**



|1.0|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
|0.0<br>0.2<br>0.4<br>0.6<br>0.8<br><br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>~~Age, SEX, BMI, KL, Surg, Inj, WOMAC (0.63 [0.61, 0.65])~~<br>Age, SEX, BMI, KL (0.61 [0.59, 0.63])<br>Age, SEX, BMI, Surg, Inj, WOMAC (0.56 [0.53, 0.58])<br>Age, SEX, BMI (0.51 [0.49, 0.54])|||||
|0.0<br>0.2<br>0.4<br>0.6<br>0.8<br><br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>~~Age, SEX, BMI, KL, Surg, Inj, WOMAC (0.63 [0.61, 0.65])~~<br>Age, SEX, BMI, KL (0.61 [0.59, 0.63])<br>Age, SEX, BMI, Surg, Inj, WOMAC (0.56 [0.53, 0.58])<br>Age, SEX, BMI (0.51 [0.49, 0.54])|||||
|0.0<br>0.2<br>0.4<br>0.6<br>0.8<br><br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>~~Age, SEX, BMI, KL, Surg, Inj, WOMAC (0.63 [0.61, 0.65])~~<br>Age, SEX, BMI, KL (0.61 [0.59, 0.63])<br>Age, SEX, BMI, Surg, Inj, WOMAC (0.56 [0.53, 0.58])<br>Age, SEX, BMI (0.51 [0.49, 0.54])|||||
|0.0<br>0.2<br>0.4<br>0.6<br>0.8<br><br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>~~Age, SEX, BMI, KL, Surg, Inj, WOMAC (0.63 [0.61, 0.65])~~<br>Age, SEX, BMI, KL (0.61 [0.59, 0.63])<br>Age, SEX, BMI, Surg, Inj, WOMAC (0.56 [0.53, 0.58])<br>Age, SEX, BMI (0.51 [0.49, 0.54])|||||
|0.0<br>0.2<br>0.4<br>0.6<br>0.8<br><br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>~~Age, SEX, BMI, KL, Surg, Inj, WOMAC (0.63 [0.61, 0.65])~~<br>Age, SEX, BMI, KL (0.61 [0.59, 0.63])<br>Age, SEX, BMI, Surg, Inj, WOMAC (0.56 [0.53, 0.58])<br>Age, SEX, BMI (0.51 [0.49, 0.54])|~~Age, SEX, BMI, K~~<br>Age, SEX, BMI, KL<br>Age, SEX, BMI, Su<br>Age, SEX, BMI (0.|~~, Surg, Inj, W~~<br> (0.61 [0.59, 0<br>rg, Inj, WOMA<br>51 [0.49, 0.54|~~MAC (0.63 ~~<br>.63])<br>C (0.56 [0.5<br>])|~~0.61, 0.65]~~<br>3, 0.58])|
|0.0<br>0.2<br>0.4<br>0.6<br>0.8<br><br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>~~Age, SEX, BMI, KL, Surg, Inj, WOMAC (0.63 [0.61, 0.65])~~<br>Age, SEX, BMI, KL (0.61 [0.59, 0.63])<br>Age, SEX, BMI, Surg, Inj, WOMAC (0.56 [0.53, 0.58])<br>Age, SEX, BMI (0.51 [0.49, 0.54])|||||


**(b)**



**Figure 3.** Assessment of Gradient Boosting Machine-based models’ performance. The subplot (a) demonstrates the ROC
curves and the subplot (b) precision-recall curves. Black dashed lines indicate the performance of a random classifier in case of
AUC, and performance of the prediction model based on the dataset labels distribution. The subplots’ legends reflect the
benchmarked models and the values of corresponding metrics with 95% confidence intervals. Here, Area under the ROC curve
metric is used in subplot (a) and Average Precision in subplot (b).


cross-validation experiment on the training set. On the test set, the CNN yielded AUC of 0.79 (0.77-0.80) and AP of 0.68
(0.66-0.70). We compared this model to the strongest reference method – model 4, and also strongest conventional method
based on LR – model 2 (Figure 4). We obtained a statistically significant performance difference in AUC (DeLong’s _p_ -value
_<_ 1 _e_ _−_ 5) when compared our CNN to the model 4.
To gain insight into the basis of the CNN’s prediction, we used the GradCAM [24] approach and visualized the attention maps
for the well-predicted knees. Examples of attention maps are presented in Figure 5. We observed that in various cases, the
CNN paid attention to the compartment opposite to the one where degenerative change became visible during the follow-up
visits. Additional examples of such attention maps are presented in Supplementary Figures 3, 4, 5 and 6.

To evaluate whether a combination of conventional diagnostic measures used in models 1-4 and CNN would further increase
the predictive accuracy, we utilized a GBM in a stacked generalization fashion [25] and treated both clinical measures and CNN’s
predictions as input features for the GBM (see Figure 1). Two stacked models were created. The first model, model 6, is
fully automatic (does not use a KL-grade as an input) and predicts a probability of OA progression. It was built using all the
predictions produced by the CNN – _P_ ( _KL_ = _i|x_ ) for _i ∈{_ 0 _,...,_ 3 _}_ and _P_ ( _y_ = _i|x_ ) for _i ∈{_ 0 _,...,_ 2 _}_, and additionally age, sex,
BMI, knee injury history, knee surgery history and WOMAC total score. The second model, model 7, was similar to the model
6, but with the addition of the KL grade that provides additional source information about the current stage of OA to the GBM.
More details on building and training this two-stage pipeline are given in Methods. We hypothesized that a radiologist and a
neural network may assign a KL grade differently, therefore, the difference in gradings could be leveraged for the prediction
model, _e.g._ if these gradings differ.

Figure 6 shows the ROC and PR curves of models 6 and 7, along with the best reference method, model 4. As reported
earlier, this reference model yielded AUC of 0.76 (0.75-0.78) and AP of 0.63 (0.61-0.65). In contrast, our multi-modal methods
without and with utilization of a KL grade – model 6 and model 7, yielded AUC of 0.79 (0.78-0.81), AP of 0.68 (0.66-0.71)
and AUC of 0.80 (0.79-0.82), AP of 0.70 (0.68-0.72) respectively. Additionally, we also show the ROC and PR curves for
model 2 in Figure 6. In Table 2, we present a detailed comparison of models 2, 4, 5, 6 and 7.

Finally, we also present the results on predicting OA progression for the subgroup of knees identified as KL-0 or KL-1
at baseline. These results are presented in Table 3. The results for this particular group of knees show that our method is
capable of identifying knees that will progress to OA in a fully automatic manner with high performance – our two best models,


**5/20**


|1.0|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
|0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>CNN (0.79 [0.78, 0.8])<br>GBM ref (0.76 [0.75, 0.78])<br>LR ref. (0.75 [0.74, 0.77])|||||||
|0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>CNN (0.79 [0.78, 0.8])<br>GBM ref (0.76 [0.75, 0.78])<br>LR ref. (0.75 [0.74, 0.77])|||||||
|0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>CNN (0.79 [0.78, 0.8])<br>GBM ref (0.76 [0.75, 0.78])<br>LR ref. (0.75 [0.74, 0.77])|||||||
|0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>CNN (0.79 [0.78, 0.8])<br>GBM ref (0.76 [0.75, 0.78])<br>LR ref. (0.75 [0.74, 0.77])|||||||
|0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>CNN (0.79 [0.78, 0.8])<br>GBM ref (0.76 [0.75, 0.78])<br>LR ref. (0.75 [0.74, 0.77])|||CNN<br>GBM<br>LR r|CNN<br>GBM<br>LR r||, 0.8])<br>.75, 0.78])<br>74, 0.77])|
|0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>CNN (0.79 [0.78, 0.8])<br>GBM ref (0.76 [0.75, 0.78])<br>LR ref. (0.75 [0.74, 0.77])|||CNN<br>GBM<br>LR r|CNN<br>GBM<br>LR r|(0.79 [0.78<br> ref (0.76 [0<br>ef. (0.75 [0.|(0.79 [0.78<br> ref (0.76 [0<br>ef. (0.75 [0.|
|0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>CNN (0.79 [0.78, 0.8])<br>GBM ref (0.76 [0.75, 0.78])<br>LR ref. (0.75 [0.74, 0.77])|||CNN<br>GBM<br>LR r|CNN<br>GBM<br>LR r|||


**(a)**



|GBM ref (0.63 [0.61, 0.65])<br>LR ref. (0.62 [0.6, 0.64])<br>0.8<br>0.6<br>0.4<br>0.2<br>0.0<br>0.0 0.2 0.4 0.6 0.8 1|Col2|GB<br>LR|M ref (0.63 [0<br>ref. (0.62 [0.6|.61, 0.65])<br>, 0.64])|
|---|---|---|---|---|
|0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br><br>GBM ref (0.63 [0.61, 0.65])<br>LR ref. (0.62 [0.6, 0.64])|||||
|0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br><br>GBM ref (0.63 [0.61, 0.65])<br>LR ref. (0.62 [0.6, 0.64])|||||
|0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br><br>GBM ref (0.63 [0.61, 0.65])<br>LR ref. (0.62 [0.6, 0.64])|||||
|0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br><br>GBM ref (0.63 [0.61, 0.65])<br>LR ref. (0.62 [0.6, 0.64])|||||


**(b)**



**Figure 4.** Comparison of the deep convolutional neural network (CNN) and the reference methods built using Gradient
Boosting Machine (GBM). Reference method based on Logistic Regression is also presented for better visual comparison
(model 2 in the text). CNN model utilizes solely knee image and the GBM model utilizes KL grade and clinical data (model 4
in the text). Subplot (a) shows the ROC curves for CNN and GBM respectively. Subplot (b) shows the Precision-Recall Curves.
Black dashed lines indicate the performance of a random classifier in case of AUC, and performance of the prediction model
based on the dataset labels distribution. The subplots’ legends reflect the benchmarked models and the values of corresponding
metrics with 95% confidence intervals. Here, Area under the ROC curve metric is used in subplot (a) and Average Precision in
subplot (b).


**(a)** **(b)** **(c)** **(d)**


**Figure 5.** Examples of attention maps for progression cases and the corresponding visualization of progression derived using
follow-up images from MOST datasets. Here, subplots (a) and (c) show the attention maps derived using a GradCAM approach.
Subplots (b) and (d) show the joint-space areas from all the follow-up images (baseline to 84 months). Here, the subplot (b)
corresponds to the attention map a) and the subplot (d) corresponds to the attention map (c).


model 6 and model 7, yielded AUC of 0.78 (0.76-0.80) and 0.80 (0.78-0.82) respectively, and AP of 0.58 (0.55-0.62) and 0.62
(0.58-0.65) respectively.


**6/20**


|1.0|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
|0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>~~Stacking w. KL (0.81 [0.79, 0.82])~~<br>Stacking w/o KL (0.79 [0.78, 0.81])<br>GBM ref (0.76 [0.75, 0.78])<br>LR ref. (0.75 [0.74, 0.77])|||||||
|0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>~~Stacking w. KL (0.81 [0.79, 0.82])~~<br>Stacking w/o KL (0.79 [0.78, 0.81])<br>GBM ref (0.76 [0.75, 0.78])<br>LR ref. (0.75 [0.74, 0.77])|||||||
|0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>~~Stacking w. KL (0.81 [0.79, 0.82])~~<br>Stacking w/o KL (0.79 [0.78, 0.81])<br>GBM ref (0.76 [0.75, 0.78])<br>LR ref. (0.75 [0.74, 0.77])|||||||
|0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>~~Stacking w. KL (0.81 [0.79, 0.82])~~<br>Stacking w/o KL (0.79 [0.78, 0.81])<br>GBM ref (0.76 [0.75, 0.78])<br>LR ref. (0.75 [0.74, 0.77])||||~~Stacking w. ~~<br>Stacking w/o<br>GBM ref (0.7<br>LR ref. (0.75|~~L (0.81 [0.~~<br> KL (0.79 [0<br>6 [0.75, 0.7<br> [0.74, 0.77|~~9, 0.82])~~<br>.78, 0.81])<br>8])<br>])|
|0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>~~Stacking w. KL (0.81 [0.79, 0.82])~~<br>Stacking w/o KL (0.79 [0.78, 0.81])<br>GBM ref (0.76 [0.75, 0.78])<br>LR ref. (0.75 [0.74, 0.77])|||||||
|0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>~~Stacking w. KL (0.81 [0.79, 0.82])~~<br>Stacking w/o KL (0.79 [0.78, 0.81])<br>GBM ref (0.76 [0.75, 0.78])<br>LR ref. (0.75 [0.74, 0.77])|||||||
|0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>~~Stacking w. KL (0.81 [0.79, 0.82])~~<br>Stacking w/o KL (0.79 [0.78, 0.81])<br>GBM ref (0.76 [0.75, 0.78])<br>LR ref. (0.75 [0.74, 0.77])|||||||
|0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>~~Stacking w. KL (0.81 [0.79, 0.82])~~<br>Stacking w/o KL (0.79 [0.78, 0.81])<br>GBM ref (0.76 [0.75, 0.78])<br>LR ref. (0.75 [0.74, 0.77])|||||||


**(a)**



|Stacking w/o KL (0.68 [0.66, 0.7])<br>GBM ref (0.63 [0.61, 0.65])<br>LR ref. (0.62 [0.6, 0.64])<br>0.8<br>0.6<br>0.4<br>0.2<br>0.0<br>0.0 0.2 0.4 0.6 0.8|Col2|Stacking<br>GBM ref (<br>LR ref. (0.|w/o KL (0.68 [<br>0.63 [0.61, 0.<br>62 [0.6, 0.64|0.66, 0.7])<br>65])<br>])|
|---|---|---|---|---|
|0.0<br>0.2<br>0.4<br>0.6<br>0.8<br><br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br><br>Stacking w/o KL (0.68 [0.66, 0.7])<br>GBM ref (0.63 [0.61, 0.65])<br>LR ref. (0.62 [0.6, 0.64])|||||
|0.0<br>0.2<br>0.4<br>0.6<br>0.8<br><br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br><br>Stacking w/o KL (0.68 [0.66, 0.7])<br>GBM ref (0.63 [0.61, 0.65])<br>LR ref. (0.62 [0.6, 0.64])|||||
|0.0<br>0.2<br>0.4<br>0.6<br>0.8<br><br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br><br>Stacking w/o KL (0.68 [0.66, 0.7])<br>GBM ref (0.63 [0.61, 0.65])<br>LR ref. (0.62 [0.6, 0.64])|||||


**(b)**



**Figure 6.** Comparison of the multi-modal methods, based on Deep Convolutional Neural Network (CNN) and Gradient
Boosting Machine (GBM) classifier versus the strongest reference method (model 4). Reference method based on Logistic
Regression is also presented for better visual comparison (model 2). The subplots’ legends reflect the benchmarked models and
the values of corresponding metrics with confidence intervals. Black dashed lines indicate the performance of a random
classifier in case of AUC, and performance of the prediction model based on the dataset labels distribution. Here, Area under
the ROC curve is used in subplot a) and Average Precision in subplot (b). The subplots (a) and (b) show the ROC and
Precision-Recall (PR) curves respectively. The results in this plot indicate that our method benefits from the utilization of a
KL-grade.


**Discussion**


In this study, we presented a patient-specific machine learning-based method to predict structural knee OA progression from
patient data acquired at a single clinical visit. The key difference of our method to the prior work is that it leverages the raw
image of the patient’s knee instead of any measures derived by human observers (e.g. JSW, KL or texture descriptors).
The results presented in this study demonstrate that our method yields significantly better prediction performance than the
conventionally used reference methods. The major finding of this study is that it is possible to predict knee OA progression from
a single knee radiograph complemented with clinical data in a fully automatic manner. Other findings of this study demonstrate
that the knee X-ray image alone is already a very powerful source of data to predict whether a particular knee will have OA
progression or not. Finally, one of the main results from a clinical point of view is that it is possible to predict progression for
patients having KL-0 and KL-1 at baseline.
To the best of our knowledge, this is the first study where CNNs were utilized to predict OA progression directly from
radiographs, and it is also one of the few studies in the field where an independent test set is used to robustly assess the
results [10][,] [14][,] [15] . We believe that having such settings, where the test set remains unused until the final model’s validation, is
crucial for further development of the OA progression prediction models. Another novelty of our approach is leveraging
multi-modal patient data: plain radiographs (raw image data compared to KL-grades used previously [10][,] [15] or manually designed
texture parameters [11][,] [12] ), symptomatic assessment, and patient’s injury and/or surgery history data for prediction. Our results
highlight that a combination of all the data allows to make more accurate predictions. Furthermore, thanks to GBM, with this
approach it was possible to use missing data without imputation.
In principle, clinical application of the developed method is straightforward and makes it possible to detect OA progression
at a low cost in primary health care with minimal modifications to the current diagnostic chain. Our method can be utilized in a
fully-automatic manner without a radiologist’s statement, and therefore, it could become available as an e.g. cloud service or
software for physiotherapists to design behavioral interventions for the cases having high confidence of prediction. Compared
to the other imaging modalities, such as MRI, the progression prediction methods developed just using radiographs and other


**7/20**


**Table 2.** Detailed comparison of the developed models for all subjects included into testing conducted on the MOST dataset.
95% confidence intervals are reported in parentheses for each of the reported metric.


**Model #** **Model** **AUC** **AP**


Age, Sex, BMI, Injury,
2 0.75 (0.74-0.77) 0.62 (0.60-0.64)
Surgery, WOMAC, KL-grade (LR)


Age, Sex, BMI, Injury,
4 0.76 (0.75-0.78) 0.63 (0.61-0.65)
Surgery, WOMAC, KL-grade (GBM)


5 CNN 0.79 (0.77-0.80) 0.68 (0.66-0.70)


CNN + Age, Sex, BMI, Injury,
6 0.79 (0.78-0.81) 0.68 (0.66-0.71)
Surgery, WOMAC (GBM-based fusion)


7 CNN + Age, Sex, BMI, Injury, 0.80 (0.79-0.82) 0.70 (0.68-0.72)
Surgery, WOMAC, KL-grade (GBM-based fusion)


KL-grade – Kellgren-Lawrence grade
CNN – Deep Convolutional Neural Network
BMI – Body-Mass Index
WOMAC – Western Ontario and McMaster Universities Arthritis Index

AUC – Area Under the Receiver Operating Characteristic Curve
AP – Average Precision
LR – Logistic regression
GBM – Gradient Boosting Machine


**Table 3.** Detailed comparison of the developed models for knees identified with Kellgren-Lawrence grade 0 or 1, which is
considered as absence of osteoarthritis. The testing was done on the Multicenter Osteoarthritis Study dataset. 95% confidence
intervals are reported in parentheses for each of the reported metric.


**Model #** **Model** **AUC** **AP**


Age, Sex, BMI, Injury,
2 0.73 (0.70-0.75) 0.52 (0.49-0.55)
Surgery, WOMAC, KL-grade (LR)


Age, Sex, BMI, Injury,
4 0.75 (0.72-0.77) 0.54 (0.51-0.58)
Surgery, WOMAC, KL-grade (GBM)


5 CNN 0.78 (0.76-0.80) 0.58 (0.55-0.61)


CNN + Age, Sex, BMI, Injury,
6 0.78 (0.76-0.80) 0.58 (0.55-0.62)
Surgery, WOMAC (GBM-based fusion)


7 CNN + Age, Sex, BMI, Injury, 0.80 (0.78-0.82) 0.62 (0.58-0.65)
Surgery, WOMAC, KL-grade (GBM-based fusion)


KL-grade – Kellgren-Lawrence grade
CNN – Deep Convolutional Neural Network
BMI – Body-Mass Index
WOMAC – Western Ontario and McMaster Universities Arthritis Index

AUC – Area Under the Receiver Operating Characteristic Curve
AP – Average Precision
LR – Logistic regression
GBM – Gradient Boosting Machine


easily obtainable data utilized in our study have potential to be the most accessible worldwide.
While machine learning-based approaches yield stronger prediction than conventional statistical models, ( _e.g._ LR), they
are less transparent, which can lead to lack of trust from clinicians. To address this drawback, various methods have been


**8/20**


developed to explain the decisions of ”black-box systems” [24][,] [26][,] [27] . As such, we utilized the GradCAM approach [24] that allowed
us generating an attention map, in order to highlight the zones where the CNN has paid its attention. While being attractive,
this approach can also lead to wrong interpretations, i.e. there is no theoretical guarantee that the neural network identifies
causal relationships between image features and the output variable. Therefore, a thorough analysis of the attention maps is
required to assess the significance of certain features and anatomical zones picked-up by the model. Such analysis, however,
could enable new possibilities for investigation of the visual features. For example, we observed interesting associations in
the GradCAM-generated attention maps (Figure 5), some of which are not captured by KL grading. As such, tibial spines
(previously associated with OA progression [28] ) were highlighted in multiple attention maps. These associations, however, do
not hold for all the progressors.
Although our study demonstrates a novel method, which outperforms various state-of-the art reference approaches, it
also has several important limitations. Firstly, our model has not been tested in other populations than the ones from the
United States. Testing the developed model on data from other populations would be a crucial step to bring the developed
machine learning-based approach to primary healthcare. Secondly, we utilized only standardized radiographs acquired with a
positioning frame, which is not used in all the hospitals worldwide. Therefore, a validation of our model using the images
acquired without the positioning frame is still needed. However, we tried to address this limitation by including data acquired
under different beam angles to the test set. Thirdly, we relied only on the KL-grading system to define a progression outcome,
and the symptomatic component of OA progression was completely ignored. This also needs to be addressed in the future
studies. Finally, we used imputation in the test set when evaluating LR models. This could potentially lower the performance
of LR-based reference methods. In contrast, GBM-based approach allowed us to leverage all the samples with missing data
without imputation.
The results presented in this study show that, for subjects at risk, our proposed knee OA progression prediction model allows
to identify the progressor cases on average 6% more accurately than with the methods previously used in the OA literature.
This study is an important step towards speeding up the OA disease modifying drug development process and also towards the
development of better personalized treatment plans.


**Methods**


**Data description and pre-processing**
We utilized Osteoarthritis Initiative (OAI, [https://data-archive.nimh.nih.gov/oai](https://data-archive.nimh.nih.gov/oai) ) and Multicenter Osteoarthritis Study (MOST, [http://most.ucsf.edu](http://most.ucsf.edu) ) follow-up cohorts. Both OAI and MOST datasets include clinical and imaging
data from subjects at risk of developing OA 45-79 and 50-79 years old, from baseline to 96 (9 imaging follow-ups) and 84
months (4 imaging follow-ups), respectively. OAI dataset includes bilateral posterior-anterior knee images, acquired with a
Synaflexer [TM] frame [29] and 10 degrees beam angle, while the MOST dataset also has images acquired with 5- and 15-degrees
beam angles.
Our inclusion criteria were the following. Firstly, we excluded the knees that had TKA, end-stage OA (KL-4) or had a
missing KL-data at the baseline. Subsequently, we excluded the knees which did not progress and were not examined at the
last follow-up. This allowed us to ensure that the subjects in the train and test sets did not progress within 96 and 84 months,
respectively. If the knee had any increase of the KL-grade during the follow-up, we assigned the class of the earliest noticed
KL-grade increase, e.g. if the knee progressed at 30 months and 84 months, we used 30-months follow-up visit to define the
fine-grained progression class. Data selection flowcharts for OAI and MOST datasets are presented in Supplementary Figures 1
and 2, respectively. The exact implementation of this selection process is also presented in the supplied source code (see Data
Availability Statement).
In our experiments, we utilized variables such as age, sex, BMI, injury history, surgery history and total WOMAC (Western
Ontario and McMaster Universities Arthritis Index) score. Due to the presence of missing values, it would be impossible to
train and test LR model without utilizing imputation techniques or removing the missing data. Therefore, during the training of
LR, we excluded the knees with missing values. In the test dataset (MOST), we imputed the missing variables by utilizing
mean value imputation strategy when testing the LR. When we trained GBM-based method, the imputation strategies are not
needed, thus we used the data extracted from MOST metadata as is.


**Image pre-processing**
To pre-process the OAI and MOST DICOM images, for each knee we extracted a region of interest (ROI) of 140 _×_ 140 mm
using an ad-hoc script and BoneFinder software [30] that enables accurate landmark localization using regression voting approach.
This was done in order to standardize the coordinate frame among the patients and the data acquisition centers. After localizing
the bone landmarks, we rotated all the knee images so that the tibial plateau was horizontal. Subsequently, we performed a
histogram clipping between 5 _[th]_ and 99 _[th]_ percentiles and used global contrast normalization subtracting the image minimum
and dividing all the image pixels by the maximum pixel value. Then, we converted the images to 8-bit depth multiplying them


**9/20**


by 255. Finally, all the images were resized to 310 _×_ 310 pixels (new pixel spacing of 0.45 mm) and the left knee images were
flipped horizontally to match the collateral (right) knee.


**Experimental setup and reference methods**
All experiments, including the hyper-parameter search, were carried out using the same 5-fold subject-wise cross-validation on
OAI data. A stratified cross-validation was used to obtain the same distribution of progressed and non-progressed cases in
both train and validation splits for each fold. To implement this validation scheme, we used the publicly available scikit-learn
package [31] .
For building regularized LR models, we used scikit-learn and for non-regularized LR we used the statsmodels package [32] .
For GBM models, we utilized the LightGBM [33] implementation. We built the CNN models using PyTorch 1.0 [34] and trained
them using three NVidia GTX 1080Ti cards.
To find the best hyperparameters set for GBM, we used the Bayesian hyperparameters optimization package hyperopt [35]

with 500 trials. Each trial maximized the AP on cross-validation. In the case of CNN, we also used cross-validation and built
5 models. We used the snapshot of the model’s weights that yielded the maximum AP value on the validation set in each
cross-validation split. The hyperparameters for CNN were found empirically.


**Deep neural network’s implementation details**
We designed a multi-task CNN architecture to predict OA progression, and our model consisted of a convolutional (Conv) and
two fully-connected (FC) blocks. One FC layer had three outputs corresponding to the three progression classes, and the other
had 5 outputs, corresponding to the prediction of the current – baseline KL grade. This is schematically illustrated in Figure 1.
To harmonize the size of the outputs after Conv layers and the inputs of the FC layers, we utilized a Global Average Pooling
layer.
We used the design of the Conv layers from se-resnext50 ~~3~~ 2x4d network [23] . In the initial cross-validation experiments, we
also evaluated se-resnet50, inceptionv4, se-resnext101 ~~3~~ 2x4d; however, we did not obtain significantly better results than the
ones reported in this study. To train the CNN, we utilized a transfer learning similarly to [7] and initialized the weights of all the
Conv layers from a network trained on the ImageNet dataset [36] . The two FC layers were initialized from random noise.
In contrast to the FC layers, the weights of the Conv layers were not trained during the first 2 epochs (full passes through
the training set) and then they were unfrozen. Subsequently, all the layers of the CNN were trained for 20 epochs. Such strategy
ensured that the FC layers did not corrupt the pre-trained Conv weights during the first backpropagation passes. The CNN was
trained with a learning rate of 1 _e_ _−_ 3 (dropped at 15 _[th]_ epoch), batch size of 64, weight decay of 1 _e_ _−_ 4 and Adam optimization
method [37] . We also placed a dropout layer [38] with the rate of _p_ = 0 _._ 5 before each FC layer.
During the training of the CNN, we used random noise addition, random rotation _±_ 5 degrees, random cropping of
the original 310 _×_ 310 pixels image to 300 _×_ 300 pixels ( 135 _×_ 135 mm) and also random gamma correction. These data
augmentations were performed randomly on-the-fly, with the aim to train our model to be invariant towards different data
acquisition parameters. We used the SOLT package of version 0.1.3 [39] in our experiments.


**Inference pipeline**
At the test phase, we averaged the outputs of all the models trained in cross-validation. Additionally, for each CNN model here,
we performed 5-crop test-time augmentation (TTA). Specifically, we cropped 4 images of 300 _×_ 300 pixels from the corners
of the original image, and one same-sized crop from the center of the image. The predictions for the 5 cropped images were
eventually averaged. Subsequently, having the TTA prediction for each cross-validation model, we averaged their results as
well. This approach allowed us to reduce the variance of the CNNs and boost the prediction accuracy.
It is worth to mention that during the evaluation of CNN model alone, instead of using the fine-grained division into
progression classes, we used the probability of progression _P_ ( _prog|x_ ) as a sum of _P_ ( _y_ = 1 _|x_ ) and _P_ ( _y_ = 2 _|x_ ) . A similar
technique was previously utilized in a skin cancer prediction study [40] .


**Interpreting neural network’s decisions**
In this study, we focused not only on producing the first state-of-the-art model for knee OA progression prediction, but also
developed an approach to examine the network’s decision to assess the radiological features detected by the network. Similar to
our previous study [7], we modified the GradCAM method [24] to operate with TTA. The output of the GradCAM is an attention
map, showing which region of the image positively correlates with the output of the network.
In the previous section, we described a TTA-approach and it should be noted that all the operations including the sum of the
progression probabilities are fully differentiable, thus the application of the GradCAM here is fairly straightforward.


**10/20**


**Model stacking: fusing heterogeneous data using tree gradient boosting**
We fused the predictions of the neural network – KL grade and progression probabilities _P_ ( _KL_ = _i|x_ ), _i ∈{_ 0 _,...,_ 4 _}_ and
_P_ ( _y_ = _i|x_ ), _i ∈{_ 0 _,_ 1 _,_ 2 _}_ respectively – with other clinical measures such as patient’s age, sex, BMI, previous injury history,
symptomatic assessments (WOMAC) and, optionally, a KL grade. Such fusion is challenging, prone to overfitting and requires
a robust cross-validation scheme. A stacked generalization approach, proposed by Wolpert [25] allows to build multiple layers of
models and handle these issues.

Following our model inference strategy, we first trained the 5 CNN models corresponding to the 5 cross-validation trainvalidation splits. Subsequently, this allowed to perform the inference on each validation set in our cross-validation setup and,
therefore, obtain CNN predictions for the whole training set. When building the second-level GBM, we utilized the same
cross-validation split and used the predictions for each knee joint as input features, along with the other clinical measures.


**Statistical analyses**
We utilized Precision-Recall (PR) and ROC curves as the main methods to measure the performance of all the methods. PR
curve can be quantitatively summarized using the AP metric. The AP metric gives a general understanding on average positive
predictive value (PPV) of the method. PPV indicates the probability of the object predicted as positive (progressor in the case
of this study) actually being positive. The precision-recall curve has been shown to be more informative than the ROC curve
when comparing classifiers on imbalanced datasets [41] . ROC curve can quantitatively be summarized using the AUC. ROC curve
demonstrates a trade-off between the true positive rate (sensitivity) and the false positive rate (1 - specificity) of the classifier.
AUC represents the quality of ranking random positive examples over the random negative examples [42] .
To compute the AUC and AP on the test set, we used stratified bootstrapping with 2,000 iterations. The stratification
allowed us to reliably assess the confidence intervals for both AUC and AP. We assessed the statistical significance of the
difference between the models using DeLong’s test [43] .


**Data Availability Statement**
OAI and MOST datasets are publicly available datasets and can be requested at http://most.ucsf.edu/ and https://oai.epi-ucsf.org/.
The Dockerfile, source codes, pre-trained models and other relevant data are publicly available (https://github.com/MIPTOulu/OAProgression).


**Acknowledgements**
The OAI is a public-private partnership comprised of five contracts (N01- AR-2-2258; N01-AR-2-2259; N01-AR-2- 2260;
N01-AR-2-2261; N01-AR-2-2262) funded by the National Institutes of Health, a branch of the Department of Health and
Human Services, and conducted by the OAI Study Investigators. Private funding partners include Merck Research Laboratories;
Novartis Pharmaceuticals Corporation, GlaxoSmithKline; and Pfizer, Inc. Private sector funding for the OAI is managed by the
Foundation for the National Institutes of Health.

MOST is comprised of four cooperative grants (Felson - AG18820; Torner - AG18832; Lewis - AG18947; and Nevitt

- AG19069) funded by the National Institutes of Health, a branch of the Department of Health and Human Services, and
conducted by MOST study investigators. This manuscript was prepared using MOST data and does not necessarily reflect the
opinions or views of MOST investigators.
We would like to acknowledge the strategic funding of the University of Oulu, Infotech Oulu, KAUTE foundation and
Sigrid Juselius Foundation for supporting this work.
Dr. Claudia Lindner is acknowledged for providing BoneFinder.


**Author contributions**


A.T. and S.S. originated the idea of the study. A.T., S.S., and S.K. designed the study, A.T. performed the experiments and
wrote the manuscript S.K., J.T., E.R. provided the technical feedback. S.B., E.O. and J.M. provided the clinical feedback. All
authors participated in the manuscript writing and editing.


**Additional information**


**Competing financial interests**
The authors declare no competing financial interests.


**11/20**


**References**


**1.** Arden, N. & Nevitt, M. C. Osteoarthritis: epidemiology. _Best practice & research Clin. rheumatology_ **20**, 3–25 (2006).


**2.** Ferket, B. S. _et al._ Impact of total knee replacement practice: cost effectiveness analysis of data from the osteoarthritis
initiative. _bmj_ **356**, j1131 (2017).


**3.** Bedson, J., Jordan, K. & Croft, P. The prevalence and history of knee osteoarthritis in general practice: a case–control
study. _Fam. practice_ **22**, 103–108 (2005).


**4.** Jamshidi, A., Pelletier, J.-P. & Martel-Pelletier, J. Machine-learning-based patient-specific prediction models for knee
osteoarthritis. _Nat. Rev. Rheumatol._ 1 (2018).


**5.** van Oudenaarde, K. _et al._ General practitioners referring adults to mr imaging for knee pain: a randomized controlled trial
to assess cost-effectiveness. _Radiology_ **288**, 170–176 (2018).


**6.** Kellgren, J. & Lawrence, J. Radiological assessment of osteo-arthrosis. _Annals rheumatic diseases_ **16**, 494 (1957).


**7.** Tiulpin, A., Thevenot, J., Rahtu, E., Lehenkari, P. & Saarakkala, S. Automatic knee osteoarthritis diagnosis from plain
radiographs: A deep learning-based approach. _Sci. reports_ **8**, 1727 (2018).


**8.** Norman, B., Pedoia, V., Noworolski, A., Link, T. M. & Majumdar, S. Applying densely connected convolutional neural
networks for staging osteoarthritis severity from plain radiographs. _J. digital imaging_ 1–7 (2018).


**9.** Antony, J., McGuinness, K., O’Connor, N. E. & Moran, K. Quantifying radiographic knee osteoarthritis severity using
deep convolutional neural networks. In _2016 23rd International Conference on Pattern Recognition (ICPR)_, 1195–1200
(IEEE, 2016).


**10.** Kerkhof, H. J. _et al._ Prediction model for knee osteoarthritis incidence, including clinical, genetic and biochemical risk
factors. _Annals rheumatic diseases_ **73**, 2116–2121 (2014).


**11.** Janvier, T. _et al._ Subchondral tibial bone texture analysis predicts knee osteoarthritis progression: data from the osteoarthritis
initiative: tibial bone texture & knee oa progression. _Osteoarthr. cartilage_ **25**, 259–266 (2017).


**12.** Janvier, T., Jennane, R., Toumi, H. & Lespessailles, E. Subchondral tibial bone texture predicts the incidence of radiographic
knee osteoarthritis: data from the osteoarthritis initiative. _Osteoarthr. cartilage_ **25**, 2047–2054 (2017).


**13.** Kraus, V. B. _et al._ Trabecular morphometry by fractal signature analysis is a novel marker of osteoarthritis progression.
_Arthritis & Rheum. Off. J. Am. Coll. Rheumatol._ **60**, 3711–3722 (2009).


**14.** Yu, D. _et al._ Development and validation of prediction models to estimate risk of primary total hip and knee replacements
using data from the uk: two prospective open cohorts using the uk clinical practice research datalink. _Annals rheumatic_
_diseases_ **78**, 91–99 (2019).


**15.** Hosnijeh, F. S. _et al._ Development of a prediction model for future risk of radiographic hip osteoarthritis. _Osteoarthr._
_cartilage_ **26**, 540–546 (2018).


**16.** Emrani, P. S. _et al._ Joint space narrowing and kellgren–lawrence progression in knee osteoarthritis: an analytic literature
synthesis. _Osteoarthr. Cartil._ **16**, 873–882 (2008).


**17.** LaValley, M. P., McAlindon, T. E., Chaisson, C. E., Levy, D. & Felson, D. T. The validity of different definitions of
radiographic worsening for longitudinal studies of knee osteoarthritis. _J. clinical epidemiology_ **54**, 30–39 (2001).


**18.** Schmidhuber, J. Deep learning in neural networks: An overview. _Neural Networks_ **61**, 85–117, DOI: 10.1016/j.neunet.
2014.09.003 (2015). Published online 2014; based on TR arXiv:1404.7828 [cs.NE].


**19.** LeCun, Y., Bengio, Y. & Hinton, G. Deep learning. _nature_ **521**, 436 (2015).


**20.** Friedman, J. H. Greedy function approximation: a gradient boosting machine. _Annals statistics_ 1189–1232 (2001).


**21.** Friedman, J., Hastie, T. & Tibshirani, R. _The elements of statistical learning_ (Springer series in statistics New York, 2001).


**22.** Bellamy, N., Buchanan, W. W., Goldsmith, C. H., Campbell, J. & Stitt, L. W. Validation study of womac: a health status
instrument for measuring clinically important patient relevant outcomes to antirheumatic drug therapy in patients with
osteoarthritis of the hip or knee. _The J. rheumatology_ **15**, 1833–1840 (1988).


**23.** Hu, J., Shen, L. & Sun, G. Squeeze-and-excitation networks. In _Proceedings of the IEEE conference on computer vision_
_and pattern recognition_, 7132–7141 (2018).


**24.** Selvaraju, R. R. _et al._ Grad-cam: Visual explanations from deep networks via gradient-based localization. In _Proceedings_
_of the IEEE International Conference on Computer Vision_, 618–626 (2017).


**12/20**


**25.** Wolpert, D. H. Stacked generalization. _Neural networks_ **5**, 241–259 (1992).


**26.** Olah, C. _et al._ The building blocks of interpretability. _Distill_ **3**, e10 (2018).


**27.** Bach, S. _et al._ On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation. _PloS one_
**10**, e0130140 (2015).


**28.** Kinds, M. B. _et al._ Quantitative radiographic features of early knee osteoarthritis: development over 5 years and relationship
with symptoms in the check cohort. _The J. rheumatology_ **40**, 58–65 (2013).


**29.** Kothari, M. _et al._ Fixed-flexion radiography of the knee provides reproducible joint space width measurements in
osteoarthritis. _Eur. radiology_ **14**, 1568–1573 (2004).


**30.** Lindner, C., Bromiley, P. A., Ionita, M. C. & Cootes, T. F. Robust and accurate shape model matching using random forest
regression-voting. _IEEE transactions on pattern analysis machine intelligence_ **37**, 1862–1874 (2015).


**31.** Pedregosa, F. _et al._ Scikit-learn: Machine learning in python. _J. machine learning research_ **12**, 2825–2830 (2011).


**32.** Seabold, S. & Perktold, J. Statsmodels: Econometric and statistical modeling with python. In _Proceedings of the 9th_
_Python in Science Conference_, vol. 57, 61 (Scipy, 2010).


**33.** Ke, G. _et al._ Lightgbm: A highly efficient gradient boosting decision tree. In _Advances in Neural Information Processing_
_Systems_, 3146–3154 (2017).


**34.** Paszke, A. _et al._ Automatic differentiation in pytorch. In _NIPS-W_ (2017).


**35.** Bergstra, J., Yamins, D. & Cox, D. D. Hyperopt: A python library for optimizing the hyperparameters of machine learning
algorithms. In _Proceedings of the 12th Python in science conference_, 13–20 (Citeseer, 2013).


**36.** Deng, J. _et al._ Imagenet: A large-scale hierarchical image database. In _2009 IEEE conference on computer vision and_
_pattern recognition_, 248–255 (Ieee, 2009).


**37.** Kingma, D. P. & Ba, J. Adam: A method for stochastic optimization. _arXiv preprint arXiv:1412.6980_ (2014).


**38.** Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I. & Salakhutdinov, R. Dropout: a simple way to prevent neural
networks from overfitting. _The J. Mach. Learn. Res._ **15**, 1929–1958 (2014).


**39.** [Tiulpin, A. Solt: Streaming over lightweight transformations. https://github.com/MIPT-Oulu/solt (2019).](https://github.com/MIPT-Oulu/solt)


**40.** Esteva, A. _et al._ Dermatologist-level classification of skin cancer with deep neural networks. _Nature_ **542**, 115 (2017).


**41.** Saito, T. & Rehmsmeier, M. The precision-recall plot is more informative than the roc plot when evaluating binary
classifiers on imbalanced datasets. _PloS one_ **10**, e0118432 (2015).


**42.** Cortes, C. & Mohri, M. Auc optimization vs. error rate minimization. In _Advances in neural information processing_
_systems_, 313–320 (2004).


**43.** DeLong, E. R., DeLong, D. M. & Clarke-Pearson, D. L. Comparing the areas under two or more correlated receiver
operating characteristic curves: a nonparametric approach. _Biometrics_ **44**, 837–845 (1988).


**13/20**


**Supplementary data**


**Table 1.** Subject-level characteristics for subsets of Osteoarthritis Initiative (OAI) and Multicenter Osteoarthritis Study
(MOST) datasets, used in this study as train and test sets respectively.


**Dataset** **Age** **BMI** **# Females** **# Males**


OAI (Train) 61.16 _[±]_ 9.19 28.62 _[±]_ 4.84 1,552 1,159


MOST (Test) 62.50 _[±]_ 8.11 30.74 _[±]_ 5.97 1,303 826

BMI – Body Mass Index


**Table 2.** Knee-level characteristics for subsets of Osteoarthritis Initiative (OAI) and Multicenter Osteoarthritis Study (MOST)
datasets, used in this study as train and test sets respectively. KL-0 to KL4 represent Kellgren-Lawrence Grading scale of
osteoarthritis (OA) – from healthy knee to end-stage OA. Here, (P) indicates the knees, which progressed during the follow-up
visits and (NP) the ones which did not progress.


**KL-grade**
**Dataset** **Subset** **Total** **# Left** **# Right**

0 1 2 3 4


NP 2,133 702 569 193 0 3,597 1,803 1,794
OAI

P 271 466 346 248 0 1,331 654 677


NP 1,558 336 314 209 0 2,417 1,208 1,209
MOST

NP 322 387 380 412 0 1,501 716 785


**14/20**


**Figure 1.** Data selection flowchart for Osteoarthritis Initiative (OAI) dataset which was used to train the model.



**15/20**


**Figure 2.** Data selection flowchart for Multicenter Osteoarthritis Study (MOST) dataset which was used to test the model.


**16/20**


**(a)** KL-0 to KL-2, slow **(b)** KL-0 to KL-3, slow


**(c)** KL-0 to KL2, slow **(d)** KL-1 to KL-3, slow


**(e)** KL-1 to KL-2, fast **(f)** KL-1 to KL3, fast


**Figure 3.** Examples of GradCAM-based attention maps for the knees progressed from no osteoarthritis to osteoarthritis.
Fine-grained sub-types of progression are also specified. The presented images are of 140 _×_ 140 mm.


**17/20**


**(a)** KL-2 to KL-3, slow **(b)** KL-2 to KL-3, fast


**(c)** KL-3 to KL-4, slow **(d)** KL-3 to TKR


**(e)** KL-2 to KL-3, fast **(f)** KL-3 to KL-4, fast


**Figure 4.** Examples of GradCAM-based attention maps for the knees having osteoarthritis at baseline and progressed in the
future. Fine-grained sub-types of progression are also specified. The presented images are of 140 _×_ 140 mm.


**18/20**


**(a)** KL-1 **(b)** KL-0


**(c)** KL-1 **(d)** KL-0


**(e)** KL-1 **(f)** KL-1


**Figure 5.** Examples of GradCAM-based attention maps for the knees having no osteoarthritis at baseline and that did progress
within the next 7 years. Baseline Kellgren-Lawrence (KL) grades are specified. The presented images are of 140 _×_ 140 mm.


**19/20**


**(a)** KL-2 **(b)** KL-2


**(c)** KL-2 **(d)** KL-2


**(e)** KL-2 **(f)** KL-2


**Figure 6.** Examples of GradCAM-based attention maps for the knees having early osteoarthritis at the baseline and that did
not progress withing the next 7 years. Baseline Kellgren-Lawrence (KL) grades are specified. The presented images are of
140 _×_ 140 mm.


**20/20**


