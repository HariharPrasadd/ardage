## **DiCOVA Challenge: Dataset, task, and baseline system for COVID-19** **diagnosis using acoustics**

_Ananya Muguli_ _[†]_ _, Lancelot Pinto_ _[‡]_ _, Nirmala R._ _[†]_ _, Neeraj Sharma_ _[†]_ _, Prashant Krishnan_ _[†]_ _,_
_Prasanta Kumar Ghosh_ _[†]_ _, Rohit Kumar_ _[†]_ _, Shrirama Bhat_ _[∓]_ _,Srikanth Raj Chetupalli_ _[†]_ _,_
_Sriram Ganapathy_ _[†]_ _, Shreyas Ramoji_ _[†]_ _, Viral Nanda_ _[‡]_


_†_ Indian Institute of Science, Bangalore, _‡_ P. D. Hinduja Hospital, Mumbai, _∓_ KMC Hospital,
Mangalore



**Abstract**


The DiCOVA challenge aims at accelerating research in diagnosing COVID-19 using acoustics (DiCOVA), a topic at the intersection of speech and audio processing, respiratory health
diagnosis, and machine learning. This challenge is an open
call for researchers to analyze a dataset of sound recordings,
collected from COVID-19 infected and non-COVID-19 indi
viduals, for a two-class classification. These recordings were
collected via crowdsourcing from multiple countries, through a
website application. The challenge features two tracks, one focusing on cough sounds, and the other on using a collection of
breath, sustained vowel phonation, and number counting speech
recordings. In this paper, we introduce the challenge and provide a detailed description of the task, and present a baseline
system for the task.
**Index Terms** : COVID-19, acoustics, machine learning, respiratory diagnosis, healthcare


**1. Introduction**


The COVID-19 pandemic has emerged as a significant health
crisis. At the time of writing (15 _−_ June-2021), more than 175
million cases and more than 3 _._ 8 million casualties have been

reported by the World Health Organization (WHO) from about
200 countries across the world [1]. Physical distancing and implementation of wide-scale population testing have served as
key measures to contain the pandemic. The testing methods in
use can be broadly divided into molecular and antibody testing.
In molecular testing, chemical reagents are used to detect the
constituents, like nucleic acids and proteins, of the SARS-CoV2 virus in an individuals’ throat or nasal swab sample. The reverse transcription polymerase chain reaction (RT-PCR) is one
such testing method, and currently serves as a gold standard for
COVID-19 testing. However, cost of machinery, time, and expertise have limited the scalability of this method. The rapid
antigen test (RAT) is another molecular testing method which
alleviates the time limitation of RT-PCR but has high false negatives (low specificity). The swab based tests and molecular
tests also violate physical distancing between participant and
the health worker, posing a serious practical challenge. In summary, there is a need to discover alternative methodologies to
diagnose COVID-19 infection that are efficient in terms of time,
cost, and ease, allowing scalability.
The WHO [1] has maintained dry cough, breathing difficulty, chest pain, and fatigue as symptoms of the infection,


Thanks to the Department of Science and Technology, Government
of India.



_istrants (or teams)._


manifested between 2 _−_ 14 days after exposure to the virus.
This was also validated by a modeling study that analyzed data
pertaining to the symptoms reported by 7178 COVID-19 positive individuals [2]. The chest X-ray (and CT) scans of many
COVID-19 infected individuals have revealed infection in the

lungs [3], and effort is being directed to evaluate the feasibility of early diagnosis using imaging techniques. Interestingly,
respiratory medical literature suggests that sounds emanating
through coordinated release of air pressure through the lungs,
such as breathing, cough, and speech, are intricately tied to
changes in the anatomy and the physiology of the respiratory
system [4]. A lung infection can affect the inspiratory and expiratory capacity. This, in addition to the presence of cough,
can result in difficulty in vocalizing sustained phonation and/or
continuous speech [5, 6]. This has been the scientific principle based on which studies analyzing vocal sounds have shown
some success in detecting respiratory ailments, such as pertussis [7], chronic obstructive pulmonary disease (COPD) [8], and
tuberculosis [9].
Based on such biological plausibility, we hypothesize that
the evaluation of the accuracy of detecting COVID-19 using
the acoustics of respiratory sounds merits research. A success
can provide an excellent point-of-care, quick, easy to use, and
cost-effective tool to diagnose COVID-19 infection, and consequently contain COVID-19 spread. Altogether, it can supplement the molecular testing methods for COVID-19 detection or
screening. The DiCOVA Challenge [1] is designed to accelerate
research efforts along this direction by creation and release of
an acoustic signal dataset, and inviting researchers to build detection models and report performance on a blind test set. Since
its release on 04 _−_ Feb-2021, the DiCOVA Challenge has cre

1 [http://dicova2021.github.io/](http://dicova2021.github.io/)




Figure 2: _In each track, the dataset is grouped non-COVID and COVID subjects. The non-COVID subjects are either healthy, have_
_symptoms (cough/cold), or have pre-existing respiratory ailments (chronic lung disease, asthma, or pneumonia). The COVID subjects_
_are either symptomatic or asymptomatic COVID positive. The distribution of age, gender, and the splits of the development dataset is_
_also shown._



ated a widespread interest amongst researchers. We have received registration from more than 80 teams. These come from
various countries and professional affiliation (see Figure 1). In
this paper, we present an overview of the topic, tasks in the
challenge, and the baseline system.


**2. Literature Review**


Since the onset of the COVID-19 pandemic, several attempts
are being made to evaluate the potential of sound based screening (and diagnosis). These attempts [10, 11, 12, 13, 14, 15, 16,
17] have primarily focused on cough sounds, and are work in
progress. Brown et.al. [16] use cough and breathing sounds
from 141 COVID-19 patients, extract a collection of short-time
frame-level acoustic features and embeddings from a VGGish
network, and pass these through a logistic regression classifier.
An area-under-the-curve (AUC) 80% is reported. The study
by Imran et al. [15] uses sound samples from 48 COVID-19
patients, and reports a sensitivity of 94% (and 91% specificity) using a convolutional neural network (CNN) architecture, fed with mel-spectrogram features as the input. The study
by Bagad et.al. [17] uses cough samples from 376 COVID19 patients, and a CNN architecture based on ResNet18 with
short-time magnitude spectrogram as input, and reports an AUC
of 72%. Altogether, these studies are encouraging. The limitations include: ( _i_ ) a different COVID-19 patient population used
in each study, ( _iii_ ) varied evaluation methodology, ( _iii_ ) small
population size, and ( _iii_ ) lack of insight on acoustic feature
differences between healthy and COVID-19 individuals. The
DiCOVA Challenge is aimed to encourage multiple research
groups to analyze the same dataset, evaluate the system performance using fixed metrics, and facilitate obtaining benchmarks



for future system development.


**3. Dataset**


The DiCOVA Challenge dataset is derived from the Coswara
dataset [18], a crowd-sourced dataset of sound recordings
from COVID-19 positive and non-COVID-19 individuals. The
Coswara data is collected using a web-application [2], launched in
April-2020, accessible through the internet by anyone around
the globe. The volunteering subjects are advised to record their
respiratory sounds in a quiet environment. Each subject provides 9 audio recordings, namely, ( _a_ ) shallow and deep breathing (2 nos.), ( _b_ ) shallow and heavy cough (2 nos.), ( _c_ ) sustained
phonation of vowels [æ] (as in bat), [i] (as in beet), and [u] (as
in boot) (3 nos.), and ( _d_ ) fast and normal pace 1 to 20 number counting (2 nos.). The subjects also provided metadata corresponding to their current health status (includes COVID-19
status, any other respiratory ailments, and symptoms), demographic information like age and gender. From this Coswara
dataset, we have created two datasets: ( _a_ ) Track-1 dataset:
composed of cough sound recordings, and ( _b_ ) Track-2 dataset:
composed of deep breathing, vowel [i], and number counting
(normal pace) speech recordings.


**3.1. Metadata**


For the challenge, the subjects have been divided into two
groups, namely,


  - non-COVID: Subjects who are either healthy or have
symptoms such as cold or cough, or have pre-existing
respiratory ailments (asthma, pneumonia, chronic lung


2 [https://coswara.iisc.ac.in/](https://coswara.iisc.ac.in/)


disease), and confirm that they are not COVID-19 positive.

  - COVID: Subjects who confirm as COVID-19 positive,
either symptomatic and asymptomatic.

The Track-1 and Track-2 development datasets are composed
of 1040 (965 non-COVID subjects) and 990 (930 non-COVID
subjects), respectively. A breakdown of the subject population
with respect to symptoms, age group, and gender is shown in
Figure 2.


**3.2. Audio**


The Coswara data collection is via crowd-sourcing, which
means the quality of the audio files has high variability and
serves as a good representation of audio data collected in the
wild. A majority of the audio files are clean as confirmed
via informal listening. More than 90% of the collected files
have a sampling rate of 48 kHz and stored as WAV files.
For the challenge datasets, all audio recordings have been resampled to 44 _._ 1 kHz and compressed as FLAC format files.
The Track-1 audio files correspond to cough sound signals.
Each audio file is derived from one unique subject, and has
one or more cough bouts. In total, there are 1040 recordings. The average duration of recordings across subjects is
4 _._ 72(standard error _±_ 0 _._ 07) sec. The Track-2 audio files correspond to one of the three different sound categories, namely,
breathing, vowel [i], and 1 to 20 number counting. In total, there
are 3(categories) _×_ 990 (subjects) sound recordings in Track-2.
The average duration of recordings across subjects is: breath
17 _._ 72( _±_ 0 _._ 68) sec, vowel [i] 12 _._ 40( _±_ 0 _._ 17) sec, and number
counting speech 14 _._ 71( _±_ 0 _._ 11) sec.


**4. Challenge Tasks**


The DiCOVA challenge features two tracks. Below we present
the task and the instructions associated with each track. A participant can choose to participate in one or both the tracks.


**4.1. Track-1**


The goal is to use cough sound recordings from COVID-19 and
non-COVID-19 individuals for the task of COVID-19 detection.


  - The Track-1 development dataset is composed of cough
audio data from 1040 subjects. The dataset also contains lists corresponding to a 5 _−_ fold cross validation
split. The distribution of COVID and non-COVID in
these splits is shown in Figure 2(a). All participants are
required to adhere to these lists and report the average
performance over the 5 validation sets.

   - A separate blind evaluation dataset is provided to all participants. The participants are required to report their
COVID-19 detection scores as probabilities.

   - This is the primary track for the challenge. A baseline
system is provided, and an online leaderboard [3] is set up
for all participants to report and compare their perfor
mance.


**4.2. Track-2**


The goal is to use breathing, sustained phonation, and speech
sound recordings from COVID-19 and non-COVID-19 individuals for any kind of detailed analysis which can contribute towards COVID-19 detection.


3 https://competitions.codalab.org/competitions/29640#results




  - The Track-2 development dataset is composed of three
sets of sound recordings, namely, breathing, vowel [i],
and number counting, from 990 subjects.


   - The dataset also contains 5 train-validation splits. The
distribution of COVID and non-COVID in these splits is
shown in Figure 2(b).


  - The participants are encouraged to design COVID-19 detection systems using above splits.


  - This track has no baseline system and leaderboard. A
non-blind test set is provided to all participants.


Participants are free to use any other data except the publicly available Project Coswara dataset [4] for data augmentation,
transfer learning, etc.


**4.3. Performance Evaluation**


Both Track-1 and Track-2 are binary classification tasks. With
a focus on COVID detection, the performance is evaluated using the traditional detection metrics, namely, true positive (TP)
and false positive (FP) rates, over a range of decision thresholds between 0 _−_ 1 with a step-size of 0 _._ 0001. For track-1, the
participant is required to submit a COVID probability score for
every audio file (corresponding to a subject) in the blind test set.
In the evaluation, we use the probability scores to compute the
receiver operating characteristic (ROC) curve, and use the area
under the curve (AUC) to quantify the model performance. An
AUC _>_ 50% indicates a better than chance performance, and
an AUC closer to 100% indicates the ideal model performance.
We also compute the model specificity at 80% sensitivity.


**5. Baseline System**


**5.1. Data preparation**


The audio data is pre-processed by normalizing the amplitude
range to _±_ 1. Subsequently, a simple sample level sound activity
detection (SAD) is applied. This keeps any audio sample with
absolute value greater than 0 _._ 01 (and a margin of _±_ 50 msec
around it) and discards the rest of the audio samples. Further,
the initial and the final 20 msec audio samples are also discarded
to remove abrupt start and end burst due to device noise.


**5.2. Feature Extraction**


Here, 39 dimensional mel-frequency cepstral coefficients
(MFCC) [19] and the delta and delta-delta coefficients are extracted with a window of size 1024 samples and a hop of size
441 samples. The librosa python library [20] is used for the
computation.


**5.3. Model Training**


Three different classifier models are trained for the two class
classification tasks of COVID versus non-COVID detection.
The models are trained using the extracted features and a (class)
balanced loss function, separately, for each of the five training data splits. The implementation uses the scikit-learn
python library [21]. The classifier models include the following.


   - Logistic regression (LR): A logistic regression classifier
trained with an added _ℓ_ 2 penalty, regularization strength
of 0 _._ 01 and liblinear optimizer is used. The maximum number of iterations is chosen as 25.


4 [https://github.com/iiscleap/Coswara-Data](https://github.com/iiscleap/Coswara-Data)


Track 1 Track-2


Cough Breathing Vowel [i] Speech


















|Cough|Col2|Cough|Col4|Col5|
|---|---|---|---|---|
|0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br><br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br>TRUE POSITIVE RATE<br>LR, AUC=61.98%<br>MLP, AUC=69.85%<br>RF, AUC=67.45%<br>chance<br>|||||
|0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br><br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br>TRUE POSITIVE RATE<br>LR, AUC=61.98%<br>MLP, AUC=69.85%<br>RF, AUC=67.45%<br>chance<br>||LR<br>ML<br>RF<br>ch|, AUC=61.9<br>P, AUC=69.<br>, AUC=67.4<br>ance|8%<br>85%<br>5%|


|1.0<br>0.8<br>RATE<br>0.6 POSITIVE<br>0.4<br>TRUE<br>0.2 LR, AUC=60.94%<br>MLP, AUC=71.52%<br>RF, AUC=76.85%<br>0.0 chance<br>0.0 0.2 0.4 0.6 0.8 1.0|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br><br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br>TRUE POSITIVE RATE<br>LR, AUC=60.94%<br>MLP, AUC=71.52%<br>RF, AUC=76.85%<br>chance<br>|||LR<br>M<br>RF<br>ch|, AUC=60.94%<br>LP, AUC=71.52<br>, AUC=76.85%<br>ance|%|


|1.0<br>0.8<br>RATE<br>0.6 POSITIVE<br>0.4<br>TRUE<br>0.2 LR, AUC=67.71%<br>MLP, AUC=73.19%<br>RF, AUC=75.47%<br>0.0 chance<br>0.0 0.2 0.4 0.6 0.8 1.0|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br><br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br>TRUE POSITIVE RATE<br>LR, AUC=67.71%<br>MLP, AUC=73.19%<br>RF, AUC=75.47%<br>chance<br>|||LR<br>M<br>R<br>ch|, AUC=67.71%<br>LP, AUC=73.19<br>F, AUC=75.47%<br>ance|%|


|1.0<br>0.8<br>RATE<br>0.6 POSITIVE<br>0.4<br>TRUE<br>0.2 LR, AUC=61.22%<br>MLP, AUC=61.13%<br>RF, AUC=65.27%<br>0.0 chance<br>0.0 0.2 0.4 0.6 0.8 1.0|Col2|Col3|Col4|
|---|---|---|---|
|0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br><br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br>TRUE POSITIVE RATE<br>LR, AUC=61.22%<br>MLP, AUC=61.13%<br>RF, AUC=65.27%<br>chance<br>||LR, AUC=61.22%<br>MLP, AUC=61.13<br>RF, AUC=65.27%<br>chance|%|



Figure 3: _Illustration of baseline systems ROCs obtained on Track-1 and Track-2 test datasets._




   - Multi-layer perceptron (MLP): A single layer perceptron
with 25 hidden units, tanh() activation, and _ℓ_ 2 regularization penalty with a weight of 0 _._ 001 is used. The
loss function is optimised using the Adam optimizer with
an initial learning rate of 0 _._ 001. To implement balanced loss for MLP, the COVID class samples are randomly oversampled in appropriate proportion to match
the count of non-COVID class samples.


  - Random Forest (RF): The random forest classifier is
trained with 50 trees in the forest and _gini_ impurity criterion to measure the split quality.


**5.4. Model Inference and Decisions**


To obtain a classification score for an audio file: ( _i_ ) a preprocessing with amplitude normalization and SAD is done,
( _ii_ ) frame-level MFCC features are extracted, ( _iii_ ) frame-level
probability scores are computed using the trained model, and
( _iv_ ) all the frame scores are averaged to obtain a single COVID
probability score for the audio file.


**5.5. Results**


Table 1 depicts the AUCs obtained on the validation folds. For
each fold (shown in Fig. 2), the classifier is trained using the
training data and evaluated on the validation data. The average
validation AUC denotes the average over the AUCs for the five
folds. For Track-1, RF gave the best average AUC, equating to
70 _._ 69%, and this was followed by MLP (at 68 _._ 80%) and LR
(at 66 _._ 95%). For Track-2, RF gave superior performance on
breathing sound (75 _._ 17% AUC). All models performed similar
for vowel sound with an AUC close to 70%. The MLP gave a
superior performance for speech sound, with 73 _._ 57% AUC.

For evaluation on the test dataset, the COVID probability score
for each file was computed by taking the average over the score
outputs from the five validation fold models. The Track-1 blind
test dataset release contains 233 (41 COVID) cough audio files
for classification into COVID/non-COVID. For Track-1, the
LR, MLP, and RF gave 61 _._ 98%, 69 _._ 85%, and 67 _._ 45% AUCs,
respectively. The corresponding ROCs are shown in Fig. 3.

The Track-2 test dataset release contains 209 (21 COVID)
audio files for each of the three sound categories. Here, the RF
model gave a better performance than other models in all the
three sound categories. Its performance was best for breathing
(76 _._ 85% AUC) and worst for speech (65 _._ 27% AUC).



**Avg.Val**
**Track** **Sound** **Model**
(Std. Err.)
LR 66.95 (±1.74)
1 Cough MLP 68.54 (±1.65)
RF 70.69 (±1.39)
LR 60.95 (±2.17)
Breathing MLP 72.47 (±1.96)
RF 75.17 (±1.23)
LR 71.48 (±0.55)
2 Vowel [i] MLP 70.39 (±1.84)
RF 69.73 (±1.93)
LR 68.93 (±1.09)
Speech MLP 73.57 (±0.71)
RF 69.61 (±1.56)

Table 1: _The baseline system performance on the validation_
_folds._


**6. Conclusion**


The uniqueness of the dataset makes the DiCOVA challenge a
first-of-its kind in the INTERSPEECH conference. The practical and timely relevance of the task encourages a focused effort from researchers across the globe, and from diverse fields
such as respiratory sciences, speech and audio processing, and
machine learning. Along with the dataset, we also provide the
baseline system software to all the participants. We expect this
will serve as an example data processing pipeline for the participants. Further, participants are encouraged to explore different
kinds of features and models of their own choice to obtain significantly better performance compared to the baseline system.


**7. Acknowledgement**


We thank Anand Mohan for his enormous help in web design
and data collection efforts.


**8. References**


[[1] “WHO Coronavirus Disease (COVID-19) Dashboard,” https://](https://covid19.who.int/)
[covid19.who.int/, 2020, [Online; accessed 10-Feb-2021].](https://covid19.who.int/)


[2] C. Menni, A. M. Valdes, M. B. Freidin, C. H. Sudre, L. H.
Nguyen, D. A. Drew, S. Ganesh, T. Varsavsky, M. J. Cardoso,
J. S. El-Sayed Moustafa, A. Visconti, P. Hysi, R. C. E.
Bowyer, M. Mangino, M. Falchi, J. Wolf, S. Ourselin,
A. T. Chan, C. J. Steves, and T. D. Spector, “Realtime tracking of self-reported symptoms to predict potential


COVID-19,” _Nature_ _Medicine_, 2020. [Online]. Available:
[https://doi.org/10.1038/s41591-020-0916-2](https://doi.org/10.1038/s41591-020-0916-2)


[3] N. Islam, S. Ebrahimzadeh, J.-P. Salameh, S. Kazi, N. Fabiano,
L. Treanor, M. Absi, Z. Hallgrimson, M. Leeflang, L. Hooft,
C. Pol, R. Prager, S. Hare, C. Dennie, R. Spijker, J. Deeks,
J. Dinnes, K. Jenniskens, D. Korevaar, J. Cohen, A. Van den
Bruel, Y. Takwoingi, J. de Wijgert, J. Damen, J. Wang, and
M. McInnes, “Thoracic imaging tests for the diagnosis of
COVID-19,” _Cochrane Database of Systematic Reviews_, no. 3,
2021. [Online]. Available: [https://doi.org//10.1002/14651858.](https://doi.org//10.1002/14651858.CD013639.pub4)
[CD013639.pub4](https://doi.org//10.1002/14651858.CD013639.pub4)


[4] J. E. Huber and E. T. Stathopoulos, _Speech Breathing Across the_
_Life Span and in Disease_ . John Wiley & Sons, Ltd, 2015, ch. 2,
pp. 11–33.


[5] L. Lee, R. G. Loudon, B. H. Jacobson, and R. Stuebing, “Speech
breathing in patients with lung disease,” _American Review of Res-_
_piratory Disease_, vol. 147, pp. 1199–1199, 1993.


[6] A. Chang and M. P. Karnell, “Perceived phonatory effort and
phonation threshold pressure across a prolonged voice loading
task: a study of vocal fatigue,” _Journal of Voice_, vol. 18, no. 4,
pp. 454–466, 2004.


[7] R. X. A. Pramono, S. A. Imtiaz, and E. Rodriguez-Villegas, “A
cough-based algorithm for automatic diagnosis of pertussis,” _PloS_
_one_, vol. 11, no. 9, 2016.


[8] A. Windmon, M. Minakshi, P. Bharti, S. Chellappan, M. Johansson, B. A. Jenkins, and P. R. Athilingam, “Tussiswatch: A smartphone system to identify cough episodes as early symptoms of
chronic obstructive pulmonary disease and congestive heart failure,” _IEEE J. Biomedical and Health Informatics_, vol. 23, no. 4,
pp. 1566–1573, 2018.


[9] G. Botha, G. Theron, R. Warren, M. Klopper, K. Dheda,
P. Van Helden, and T. Niesler, “Detection of tuberculosis by automatic cough sound analysis,” _Physiological Measurement_, vol. 39,
no. 4, p. 045005, 2018.


[[10] “Cambridge University, UK - COVID-19 Sounds App,” https://](https://covid-19-sounds.org/en/)
[covid-19-sounds.org/en/, 2020, [Online; accessed 07-Aug-2020].](https://covid-19-sounds.org/en/)


[[11] “Cough Against COVID - Wadhwani AI Institute,” https://](https://coughagainstcovid.org/)
[coughagainstcovid.org/, 2020, [Online; accessed 07-Aug-2020].](https://coughagainstcovid.org/)


[12] “NYU Breathing Sounds for COVID-19,” [https:](https://breatheforscience.com/)
[//breatheforscience.com/,](https://breatheforscience.com/) 2020, [Online; accessed 07-Aug2020].


[[13] “EPFL Cough for COVID-19 Detection,” https://coughvid.epfl.](https://coughvid.epfl.ch/)
[ch/, 2020, [Online; accessed 07-Aug-2020].](https://coughvid.epfl.ch/)


[[14] “CMU sounds for COVID Project,” https://node.dev.cvd.lti.cmu.](https://node.dev.cvd.lti.cmu.edu/)
[edu/, 2020, [Online; accessed 07-Aug-2020].](https://node.dev.cvd.lti.cmu.edu/)


[15] A. Imran, I. Posokhova, H. N. Qureshi, U. Masood, M. S. Riaz,
K. Ali, C. N. John, M. I. Hussain, and M. Nabeel, “AI4COVID19: AI enabled preliminary diagnosis for COVID-19 from cough
samples via an app,” _Informatics in Medicine Unlocked_, vol. 20,
p. 100378, 2020.


[16] C. Brown, J. Chauhan, A. Grammenos, J. Han, A. Hasthanasombat, D. Spathis, T. Xia, P. Cicuta, and C. Mascolo,
“Exploring automatic diagnosis of covid-19 from crowdsourced
respiratory sound data,” in _Proc. 26th ACM SIGKDD In-_
_ternational_ _Conference_ _on_ _Knowledge_ _Discovery_ _&_ _Data_
_Mining_ . New York, NY, USA: Association for Computing Machinery, 2020, p. 3474–3484. [Online]. Available:
[https://doi.org/10.1145/3394486.3412865](https://doi.org/10.1145/3394486.3412865)


[17] P. Bagad, A. Dalmia, J. Doshi, A. Nagrani, P. Bhamare, A. Mahale, S. Rane, N. Agarwal, and R. Panicker, “Cough against covid:
Evidence of covid-19 signature in cough sounds,” _arXiv preprint_
_arXiv:2009.08790_, 2020.


[18] N. Sharma, P. Krishnan, R. Kumar, S. Ramoji, S. R. Chetupalli,
N. R., P. K. Ghosh, and S. Ganapathy, “Coswara – a database of
breathing, cough, and voice sounds for COVID-19 diagnosis,” in
_Proc. INTERSPEECH, ISCA_, 2020.




[19] S. Davis and P. Mermelstein, “Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences,” _IEEE Transactions on Acoustics, Speech, and Sig-_
_nal Processing_, vol. 28, no. 4, pp. 357–366, 1980.


[20] B. McFee, V. Lostanlen, A. Metsai, M. McVicar, S. Balke,
C. Thom´e, C. Raffel, F. Zalkow, A. Malek, Dana, K. Lee,
O. Nieto, J. Mason, D. Ellis, E. Battenberg, S. Seyfarth,
R. Yamamoto, K. Choi, viktorandreevichmorozov, J. Moore,
R. Bittner, S. Hidaka, Z. Wei, nullmightybofo, D. Here˜n´u,
F.-R. St¨oter, P. Friesch, A. Weiss, M. Vollrath, and T. Kim,
[“librosa/librosa: 0.8.0,” Jul. 2020. [Online]. Available: https:](https://doi.org/10.5281/zenodo.3955228)
[//doi.org/10.5281/zenodo.3955228](https://doi.org/10.5281/zenodo.3955228)


[21] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion,
O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg,
J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot,
and E. Duchesnay, “Scikit-learn: Machine learning in Python,”
_Journal of Machine Learning Research_, vol. 12, pp. 2825–2830,
2011.


