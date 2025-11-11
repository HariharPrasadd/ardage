## **Safely Entering the Deep: A Review of Verification and Validation for Machine** **Learning and a Challenge Elicitation in the Automotive Industry**

**Markus Borg** **[1]** **Cristofer Englund** **[1]** **, Krzysztof Wnuk** **[2]** **, Boris Duran** **[1]** **, Christoffer Levandowski** **[3]** **,**
**Shenjian Gao** **[2]** **, Yanwen Tan** **[2]** **, Henrik Kaijser** **[4]** **, Henrik L¨onn** **[4]** **, Jonas T¨ornqvist** **[3]**


_1_ _RISE Research Institutes of Sweden AB,_
_Scheelev¨agen 17,_
_SE-223 70 Lund, Sweden_

_E-mail: {markus.borg, cristofer.englund, boris.duran}@ri.se_

_2_ _Blekinge Institute of Technology,_
_Valhallav¨agen 1,_
_SE-371 41 Karlskrona, Sweden_

_E-mail: krzysztof.wnuk@bth.se, {shga13, yata13}@student.bth.se_

_3_ _QRTECH AB,_
_Fl¨ojelbergsgatan 1C,_
_SE-431 35 M¨olndal, Sweden_

_E-mail: {christoffer.levandowski, jonas.tornqvist}@qrtech.se_

_4_ _AB Volvo,_
_Volvo Group Trucks Technology,_
_SE-405 08 Gothenburg, Sweden_
_E-mail: {henrik.kaijser, henrik.lonn}@volvo.se_


**Abstract**


Deep Neural Networks (DNN) will emerge as a cornerstone in automotive software engineering. However, developing systems with DNNs introduces novel challenges for safety assessments. This paper reviews
the state-of-the-art in verification and validation of safety-critical systems that rely on machine learning.
Furthermore, we report from a workshop series on DNNs for perception with automotive experts in Sweden,
confirming that ISO 26262 largely contravenes the nature of DNNs. We recommend aerospace-to-automotive
knowledge transfer and systems-based safety approaches, e.g., safety cage architectures and simulated system

test cases.


_Keywords:_ deep learning, safety-critical systems, machine learning, verification and validation,
ISO 26262


**1.** **Introduction**


As an enabling technology for autonomous driving,
Deep learning Neural Networks (DNN) will emerge


_M. Borg et al. / Safely Entering the Deep_



as a cornerstone in automotive software engineering. Automotive software solutions using DNNs is
a hot topic, with new advances being reported almost weekly. Also in the academic context, several
research communities study DNNs in the automotive domain from various perspectives, e.g., applied
Machine Learning (ML) [75], software engineering [28],
safety engineering [78], and Verification & Validation
(V&V) [50] .

DNNs are used to enable _vehicle environmental_

_perception_, i.e., awareness of elements in the surrounding traffic. Successful perception is a prerequisite for autonomous features such as lane departure
detection, path/trajectory planning, vehicle tracking,
behavior analysis, and scene understanding [111] – and
a prerequisite to reach levels 3-5 as defined by SAE
International’s levels of driving automation. A wide
range of sensors have been used to collect input data
from the environment, but the most common approach is to rely on front-facing cameras [34] . In recent
years, DNNs have demonstrated their usefulness in
classifying such camera data, which in turn has enabled both perception and subsequent breakthroughs
toward autonomous driving [54] .

From an ISO 26262 safety assurance perspective, however, developing systems based on DNNs
constitutes a major paradigm shift compared to conventional systems _[∗]_ [28] . Andrej Karpathy, Director of
AI at Tesla, boldly refers to the new era as “Software 2.0” [†] . No longer do human engineers explicitly
describe all system behavior in source code, instead
DNNs are trained using enormous amounts of historical data.

DNNs have been reported to deliver superhuman
classification accuracy for specific tasks [35], but inevitably they will occasionally fail to generalize [93] .
Unfortunately, from a safety perspective, analyzing when this might happen is currently not possible due to the black-box nature of DNNs. A state
of-the-art DNN might be composed of hundreds of
millions of parameter weights, thus the methods
for V&V of DNN components must be different
compared to approaches for human readable source



code. Techniques enforced by ISO 26262 such as
source code reviews and exhaustive coverage testing
are not applicable [78] .
The contribution of this review paper is twofold.
First, we describe the state-of-the-art in V&V of
safety-critical systems that rely on ML. We survey
academic literature, partly through a reproducible
snowballing review [103], i.e., establishing a body of
literature by tracing referencing and referenced papers. Second, we elicit the most pressing challenges
when engineering safety-critical DNN components
in the automotive domain. We report from workshops with automotive experts, and we validate findings from the literature review through an industrial
survey. The research has been conducted as part of
SMILE [‡], a joint research project between RISE AB,
Volvo AB, Volvo Cars, QRTech AB, and Semcon
AB.

The rest of the paper is organized as follows: Section 2 presents safety engineering concepts within the automotive domain and introduces
the fundamentals of DNNs. Section 3 describes the

proposed research method, including four sources of
empirical evidence, and Section 4 reports our findings. Section 5 presents a synthesis targeting our
two objectives, and discusses implications for research and practice. Finally, Section 6 concludes the
paper and outlines the most promising directions for
future work. Throughout the paper, we use the notation **[PX]** to explicitly indicate publications that are
part of the snowballing literature study.


**2.** **Background**


This section first presents development of safetycritical software according to the ISO 26262
standard [44] . Second, we introduce fundamentals of
DNNs, required to understand how it could allow
vehicular perception. In the remainder of this paper,
we adhere to the following three definitions related
to safety-critical systems:


_•_ **Safety** is “freedom from unacceptable risk of
physical injury or of damage to the health of



_∗_ by _conventional systems_ we mean any system that does not have the ability to learn or improve from experience

_†_
https://medium.com/@karpathy/software-2-0-a64152b37c35

_‡_
The SMILE project: Safety analysis and verification/validation of MachIne LEarning based systems


_M. Borg et al. / Safely Entering the Deep_



people” [43]


_•_ **Robustness** is “the degree to which a component
can function correctly in the presence of invalid
inputs or stressful environmental conditions” [42]


_•_ **Reliability** is “the probability that a component
performs its required functions for a desired period of time without failure in specified environments with a desired confidence” [11]


_**2.1.**_ _**Safety Engineering in the Automotive**_

_**Domain: ISO 26262**_


Safety is not a property that can be added at the end
of the design. Instead, it must be an integral part of
the entire engineering process. To successfully engineer a safe system, a systematic safety analysis and
a methodological approach to managing risks are
required [8] . Safety analysis comprises identification
of hazards, development of approaches to eliminate
hazards or mitigate their consequences, and verification that the approaches are in place in the system.
Risk assessment is used to determine how safe a system is, and to analyze alternatives to lower the risks
in the system.
Safety has always been an important concern
in engineering, and best practices have often been
collected in governmental or industry _safety stan-_
_dards_ . Common standards provide a common vocabulary as well as a way for both internal and external safety assessment, i.e., work tasks for both
engineers working in the development organization and for independent safety assessors from certification bodies. For software-intensive systems,
the generic meta-standard IEC 61508 [43] introduces
the fundamentals of functional safety for Electrical/Electronic/Programmable Electronic (E/E/PE)
Safety-related Systems, i.e., hazards caused by
malfunctioning E/E/PE systems rather than nonfunctional considerations such as fire, radiation, and
corrosion. Several different domains have their own

adaptations of IEC 61508.
ISO 26262 [44] is the automotive derivative of

IEC 61508, organized into 10 parts, constituting a
comprehensive safety standard covering all aspects
of automotive development, production, and maintenance of safety-related systems. V&V are core ac


tivities in safety-critical development and thus discussed in detail in ISO 26262, especially in Part 4:
Product development at the system level and Part
6: Product development at the software level. The
scope of the current ISO 26262 standard is series
production passenger cars with a max gross weight
of 3,500 kg. However, the second edition of the
standard, expected in the beginning of 2019, will
broaden the scope to cover also trucks, buses, and
motorcycles.


The _automotive safety lifecycle_ (ASL) is one key
component of ISO 26262 [66], defining fundamental
concepts such as safety manager, safety plan, and
confirmation measures including safety review and
audit. The ASL describes six phases: management, development, production, operation, service,
and decommission. Assuming that a safety-critical
DNN will be considered a software unit, especially
the development phase on the software level (Part
6) mandates practices that will require special treatment. Examples include verification of software
implementation using inspections (Part 6:8.4.5) and
conventional structural code coverage metrics (Part
6:9.4.5). It is evident that certain ISO 26262 process requirements cannot apply to ML-based software units, in line with how model-based development is currently partially excluded.


Another key component of ISO 26262 is the _au-_
_tomotive safety integrity level_ (ASIL). In the beginning of the ASL development phase, a safety analysis of all critical functions of the system is conducted, with a focus on hazards. Then a risk analysis combining 1) the probability of exposure, 2)
the driver’s possible controllability, and 3) the possible severity of the outcome, results in an ASIL
between A and D. ISO 26262 enforces development and verification practices corresponding to the
ASIL, with the most rigorous practices required for
ASIL D. Functions that are not safety-critical, i.e.,
below ASIL A, are referred to as ‘QM’ as no more
than the normal quality management process is enforced.


_M. Borg et al. / Safely Entering the Deep_



_**2.2.**_ _**Deep Learning for Perception: Approaches**_
_**and Challenges**_


While there currently is a deep learning hype, there
is no doubt that the technique has produced groundbreaking results in various fields – by clever utilization of the increased processing power in the last
decade, nowadays available in inexpensive GPUs,
combined with the ever-increasing availability of
data.

Deep learning is enabled by DNNs, which are
a kind of Artificial Neural Networks (ANN). To
some extent inspired by biological connectomes,
i.e., mappings of neural connections such as in the
human brain, ANNs composed of connected layers
of neurons are designed to learn to perform classification tasks. While ANNs have been studied for
decades, significant breakthroughs came when the
increased processing power allowed adding more
and more layers of neurons – which also increased
the number of connections between neurons by orders of magnitude. The exact number of layers,
however, needed for a DNN to qualify as deep is
debatable.

A major advantage of DNNs is that the classifier
is less dependent on _feature engineering_, i.e., using
domain knowledge to (perhaps manually) identify
properties in data for ML to learn from – this is often difficult. Examples of operations used to extract
features in computer vision include: color analysis,
edge extraction, shape matching, and texture analysis. What DNNs instead introduced was an ML so
lution that learned those features directly from input
data, greatly decreasing the need for human feature
engineering. DNNs have been particularly successful in speech recognition, computer vision, and text
processing – areas in which ML results were limited by the tedious work required to extract effective
features.

In computer vision, essential for vehicular perception, the state-of-the-art is represented by a special class of DNNs known as _Convolutional Neural_
_Networks_ (CNN) [36] _[,]_ [95] _[,]_ [88] _[,]_ [27] . Since 2010, several approaches based on CNNs have been proposed – and
in only five years of incremental research the best
CNNs matched the image classification accuracy
of humans. CNN-based image recognition is now



reaching the masses, as companies like Nvidia, Intel, etc. are now commercializing specialized hardware with automotive applications in mind such as
the Drive PX series. Success stories in the auto
motive domain include lane keeping applications for
self-driving cars [14] _[,]_ [79] .
_Generative Adversarial Networks_ (GAN) is another approach in deep learning research that is currently receiving considerable interest [30] _[,]_ [74] . In contrast to discriminative networks (what has been discussed so far) that learn boundaries between classes
in the data for the purpose of classification, a generative network can instead be used to learn the

probability of features given a specific class. Thus,
a GAN could be used to generate samples from a
learned network – which could possibly be used to
expand available training data with additional synthetic data. GANs can also be used to generate _ad-_
_versarial examples_, i.e., inputs to ML classifiers intentionally created to cause misclassification.
Finally, successful applications of DNNs rely on
the availability of large labeled datasets from which
to learn features. In many cases, such labels are
limited or does not exist at all. To maximize the

utility of the labeled data, truly hard currency for
anyone engineering ML-based systems, techniques
such as _transfer learning_ are used to adapt knowledge learned from one dataset to another domain [29] .


**3.** **Research method**


The overarching goal of the SMILE project is to
develop approaches to V&V of ML-based systems,
more specifically automotive applications relying on
DNNs. Our current paper is guided by two research
questions:


RQ1 What is the state-of-the-art in V&V of MLbased safety-critical systems?

RQ2 What are the main challenges when engineering safety-critical systems with DNN components
in the automotive domain?


Fig. 1 shows an overview of the research, divided into three sequential parts (P1-P3). Each part
concluded with a Milestone (I–III). In Fig. 1, tasks


_M. Borg et al. / Safely Entering the Deep_



driven by academia (or research institutes) are presented in the light gray area – primarily addressing
RQ1. Tasks in the darker gray area above, are primarily geared toward collecting data in the light of
RQ2, and mostly involve industry practitioners. The
darkest gray areas denote involvement of practitioners that were active in safety-critical development
but not part of the SMILE project.


Fig. 1. Overview of the SMILE project and its three milestones. The figure illustrates the joint industry/academia nature of SMILE, indicated by light gray background for tasks
driven by academia and darker gray for tasks conducted by
practitioners.


In the first part of the project (P1 in Fig. 1),
we initiated a systematic snowballing review of academic literature to map the state-of-the-art. In parallel, we organized a workshop series with domain
experts from industry with monthly meetings to also
assess the state-of-practice in the Swedish automotive industry. The literature review was seeded by
discussions from the project definition phase (a).
Later, we shared intermediate findings from the literature review at workshop #4 (b) and final results
were brought up to discussion at workshop #6 (c).
The first part of the project concluded with Milestone I: a collection of industry perspectives.
The second part of the SMILE project (P2 in
Fig. 1) involved an analysis of the identified literature (d). We extracted challenges and solution proposals from the literature, and categorized them according to a structure that inductively emerged dur


ing the process (see Section 3.1). Subsequently, we
created a questionnaire-based survey to validate our
findings and to receive input from industry practitioners beyond SMILE (e). The second phase
concluded with analyzing the survey data at Milestone II.


_M. Borg et al. / Safely Entering the Deep_


In the third part of the project (P3 in Fig. 1), we
collected all results (f), and performed a synthesis
(g). Finally, writing this article concludes the research at Milestone III.



Fig. 2 shows an overview of the SMILE project
from an evidence perspective. The collection of
empirical evidence was divided into two independent tracks resulting in four sets of evidence, reflecting the nature of the joint academia/industry
project. Furthermore, the split enabled us to balance the trade-off between rigor and relevance that
plagues applied research projects [45] .


As shown in the upper part of Figure 2, the
SMILE consortium performed (non-replicable, from
now on: “ _ad hoc_ ”) searching for related work. An
early set of papers was used to seed the systematic
search described in the next paragraph. The findings
in the body of related work (cf. A in Fig. 2) were
discussed at the workshops. The workshops served
dual purposes, they collected empirical evidence of
priorities and current needs in the Swedish automotive industry (cf. B in Fig. 2), and they validated
the relevance of the research identified through the
_ad hoc_ literature search. The upper part focused on
_maximizing industrial relevance_, at the expense of
rigor, i.e., we are certain that the findings are relevant to the Swedish automotive industry, but the
research was conducted in an _ad hoc_ fashion with

limited traceability and replicability. The right part
of Figure 2 complements the practice-oriented research of the SMILE project by a systematic literature review, adhering to an established process [103] .
The identified papers (cf. C in Fig. 2)) were systematized and the result was validated through a
questionnaire-based survey. The survey also acted
as a means to collect additional primary evidence,
as we collected practitioners’ opinions on V&V of
ML-based systems in safety-critical domains (cf. D
in Fig. 2). Thus, the lower part focused on _maximiz-_
_ing academic rigor_ .



Fig. 2. Overview of the SMILE project from an evidence
perspective. We treat the evidence as four different sets: A.
Related work and C. Snowballed literature represent secondary evidence, whereas B. Workshop findings and D.
Survey responses constitute primary evidence.


_**3.1.**_ _**The systematic review**_


Inspired by evidence-based medicine, systematic literature reviews have become a popular software engineering research method to aggregate work in a
research area. Snowballing literature reviews [103] is
an alternative to more traditional database searches

relying on carefully developed search strings, particularly suitable when the terminology used in the
area is diverse, e.g., in early stages of new research
topics. This section describes the two main phases
of the literature review: 1) paper selection and 2)
data extraction and analysis.


_3.1.1._ _Paper selection_


As safety-critical applications of DNNs in the automotive sector is still a new research topic, we decided to broaden our literature review to encompass
also other types of ML, and also to go beyond the automotive sector. We developed the following criteria: for a publication to be included in our literature
review, it should describe 1) engineering of an MLbased system 2) in the context of autonomous cyberphysical systems, and 3) the paper should address
V&V or safety analysis. Consequently, our criteria includes ML beyond neural networks and DNNs.
Our focus on autonomous cyber-physical systems
implicitly restricts our scope to safety-critical systems. Finally, we exclude papers that do not target


_M. Borg et al. / Safely Entering the Deep_



V&V or safety analysis, but instead other engineering considerations, e.g., requirements engineering,
software architecture, or implementation issues.
First, we established a _start set_ using exploratory
searching in Google Scholar and applying our inclusion criteria. By combining various search terms
related to ML, safety analysis, and V&V identified
during the project definition phase of the workshop
series (cf. a) in Fig 1), we identified 14 papers representing a diversity of authors, publishers, and publications venues, i.e., adhering to recommendations
for a feasible start set [103] . Still, the composition of
the start set is a major threat to the validity of any
snowballing literature review. Table 5 shows the papers in the start set.
Originating in the 14 papers in the start set, we
iteratively conducted backward and forward snowballing. Backward snowballing means scanning the
reference lists for additional papers to include. Forward snowballing from a paper involves adding related papers that cite the given paper. We refer to
one combined effort of backward and forward snow
balling as an _iteration_ . In each iteration, two researchers collected candidates for inclusion and two

other researchers validated the selection using the
inclusion criteria. Despite our efforts to carefully
process iterations, there is always a risk that relevant publications could not be identified by following references from our start set due to citation
patterns in the body of scientific literature, e.g., research cliques.


_3.1.2._ _Data extraction and analysis_


When the snowballing was completed, two authors
extracted publication metadata according to a predefined extraction form, e.g., publication venue and
application domain. Second, the same two authors
conducted an assessment of rigor and relevance as
recommended by Ivarsson and Gorschek [45] . Third,
they addressed RQ1 using thematic analysis [26], i.e.,
summarizing, integrating, combining, and comparing findings of primary studies to identify patterns.
Our initial plan was to classify challenges and
solution proposals in previous work using classification schemes developed by Amodei _et al._ **[P2]** and
Varshney [101], respectively. However, neither of the



two proposed categorization schemes were successful in spanning the content of the selected papers.
To better characterize the selected body of research,
we inductively created new classification schemes
for challenges and solution proposals according to a
grounded theory approach. Table 1 defines the final
categories used in our study, seven challenge categories and five solution proposal categories.


_**3.2.**_ _**The questionnaire-based survey**_


To validate the findings from the snowballed literature (cf. C. in Figure 2), we designed a webbased questionnaire to survey practitioners in safetycritical domains. Furthermore, reaching out to additional practitioners beyond the SMILE project enables us to collect more insights into challenges
related to ML-based systems in additional safetycritical contexts (cf. D. in Fig. 2). Moreover, we
used the survey to let the practitioners rate the importance of the challenges reported in the academic
literature, as well as the perceived feasibility of the
published solutions proposals.
We designed the survey instrument using Google
Forms, structured as 10 questions organized into
two sections. The first section consisted of seven
closed-end questions related to demographics of the
respondents and their organizations and three Likert items concerning high-level statements on V&V
of ML-based systems. The second section consisted
of three questions: 1) rating the importance of the
challenge categories, 2) rating how promising the
solution proposal categories are, and 3) an open-end
free-text answer requesting a comment on our main
findings and possibly adding missing aspects.
We opted for an inclusive approach and used
convenience sampling to collect responses [76], i.e., a
non-probabilistic sampling method. The target population was software and systems engineering practitioners working in safety-critical contexts, including both engineering and managerial roles, e.g., test
managers, developers, architects, safety engineers,
and product managers. The main recruitment strategy was to invite the extended SMILE network (cf.
workshops #5 and #6 in Fig. 1) and to advertise the
survey invitation in LinkedIn groups related to development of safety-critical systems. We collected


_M. Borg et al. / Safely Entering the Deep_


Table 1: Definition of categories of challenges and solution proposals for V&V of ML-based systems.


|Challenge Categories|Definitions|
|---|---|
|State-space explosion|Challenges related to the very large size of the input space.|
|Robustness|Issues related to operation in the presence of invalid inputs or stressful envi-<br>ronmental conditions.|
|Systems engineering|Challenges related to integration or co-engineering of ML-based and conven-<br>tional components.|
|Transparency|Challenges originating in the black-box nature of the ML system.|
|Requirements speciﬁcation|Problems related to specifying expectations on the learning behavior.|
|Test speciﬁcation|Issues related to designing test cases for ML-based systems, e.g., non-<br>deterministic output.|
|Adversarial attacks|Threats related to antagonistic attacks on ML-based systems, e.g., adversarial<br>examples.|
|||
|**Solution Proposal Categories**|**Deﬁnitions**|
|Formal methods|Approaches to mathematically prove that some speciﬁcation holds.|
|Control theory|Veriﬁcation of learning behavior based on automatic control and self-adaptive<br>systems.|
|Probabilistic methods|Statistical approaches such as uncertainty calculation, Bayesian analysis, and<br>conﬁdence intervals.|
|Test case design|Approaches to create effective test cases, e.g., using genetic algorithms or pro-<br>cedural generation.|
|Process guidelines|Guidelines supporting work processes, e.g., covering training data collection<br>or testing strategies.|


_M. Borg et al. / Safely Entering the Deep_



answers in 2017, from July 1 to August 31.
As a first step of the response analysis, we performed a content sanity check to identify invalid answers, e.g., nonsense or careless responses. Subsequently, we collected summary statistics of the responses and visualized it with bar charts to get a
quick overview of the data. We calculated Spearman rank correlation ( _ρ_ ) between all ordinal scale
responses, interpreting correlations as weak, moderate, and strong for _ρ >_ 0 _._ 3, _ρ >_ 0 _._ 5, and _ρ >_ 0 _._ 7,
respectively. Finally, the two open-ended questions
were coded, summarized, and validated by four of
the co-authors.


**4.** **Results and Discussion**


This section is organized according to the evidence
perspective provided in Fig. 2: A. Related work, B.
Workshop findings, C. Snowballed literature, and D)
Survey responses. As reported in Section 3, A. and
B. focus on industrial relevance, whereas C. and D.
aim at academic rigor.


_**4.1.**_ _**Related work**_


The related work section (cf. A. in Fig. 2) presents
an overview of literature that was identified during
the SMILE project. Fourteen of the papers were selected early to seed the (independent) snowballing
literature review described in Section 3.1. In this

section, we first describe the start set **[P1]** - **[P14]**,
and then papers that were subsequently identified by
SMILE members or the anonymous reviewers of the
manuscript – but not through the snowballing process (as these are reported separately in Section 4.3).


_4.1.1._ _The snowballing start set_


The following 14 papers were selected as the snowballing start set, representing a diverse set of authors, publication venues, and publication years. We
briefly describe them below, and motivate their inclusion in the start set.


**[P1]** Clark _et al._ reported from a US Air Force
research project on challenges in V&V of autonomous systems. This work is highly related to
the SMILE project.




**[P2]** Amodei _et al._ listed five challenges to artificial
intelligence safety according to Google Brain: 1)
avoiding negative side effects, 2) avoiding reward
hacking, 3) scalable oversight, 4) safe exploration,
and 5) robustness to distributional shift.

**[P3]** Brat and Jonsson discussed challenges in V&V
of autonomous systems engineered for space exploration. Included to cover the space domain.

**[P4]** Broggi _et al._ presented extensive testing of the
BRAiVE autonomous vehicle prototype by driving from Italy to China. Included as it is different,
i.e., reporting experiences from a practical trip.

**[P5]** Taylor _et al._ sampled research in progress
(in 2003) on V&V of neural networks, aimed
at NASA applications. Included to snowball research conducted in the beginning of the millennium.

**[P6]** Taylor _et al._ with the Machine Intelligence Research Institute surveyed design principles that
could ensure that systems behave in line with the
interests of their operators – which they refer to
as “AI alignment”. Included to bring in a more
philosophical perspective on safety.

**[P7]** Carvalho _et al._ presented a decade of research
on control design methods for systematic handling
of uncertain forecasts for autonomous vehicles.

Included to cover robotics.

**[P8]** Ramos _et al._ proposed a DNN-based obstacle
detection framework, providing sensor fusion for
detection of small road hazards. Included as the

work closely resembles the use case discussed at
the workshops (see Section 4.2).

**[P9]** Alexander _et al._ suggested “situation coverage
methods” for autonomous robots to support testing of all environmental circumstances. Included
to cover coverage.

**[P10]** Zou _et al._ discussed safety assessments of
probabilistic airborne collision avoidance systems
and proposes a genetic algorithm to search for undesired situations. Included to cover probabilistic
approaches.

**[P11]** Zou _et al._ presented a safety validation approach for avoidance systems in unmanned aerial
vehicles, using evolutionary search to guide simulations to potential conflict situations in large state


_M. Borg et al. / Safely Entering the Deep_



spaces. Although the authors overlap, included to
snowball research on simulation.

**[P12]** Arnold and Alexander proposed using procedural content generation to create challenging environmental situations when testing autonomous
robot control algorithms in simulations. Included
to cover synthetic test data.

**[P13]** Sivaraman and Trivedi compared three active
learning approaches for on-road vehicle detection.
Included to add a semi-supervised ML approach.

**[P14]** Mozaffari _et al._ developed a robust safetyoriented autonomous cruise controller based on

the model predictive control technique. Included
to identify approaches based on control theory.


In the start set, we consider **[P1]** to be the
research endeavor closest to our current study.
While we target the automotive domain rather than
aerospace, both studies address highly similar research objectives – and also the method used to
explore the topic is close to our approach. **[P1]**
describes a year-long study aimed at: 1) understanding the unique challenges to the certification
of safety-critical autonomous systems and 2) identifying the V&V approaches needed to overcome
them. To accomplish this, the US Air Force organized three workshops with representatives from
industry, academia, and governmental agencies, respectively. **[P1]** concludes that that there are four
enduring problems that must be addressed:


_•_ State-Space Explosion – In an autonomous system, the decision space is non-deterministic and
the system might be continuously learning. Thus,
over time, there may be several output signals for
each input signal. This in turn makes it inherently
challenging to exhaustively search, examine, and
test the entire decision space.


_•_ Unpredictable Environments – Conventional systems have limited ability to adapt to unanticipated events, but an autonomous systems should
respond to situations that were not programmed at
design time. However, there is a trade-off between
performance and correct behavior, which exacerbates the state-space explosion problem.


_•_ Emergent Behavior – Non-deterministic and
adaptive systems may induce behavior that result



in unintended consequences. Challenges comprise how to understand all intended and unintended behavior and how to design experiments
and test vectors that are applicable to adaptive decision making in an unpredictable environment.


_•_ Human-Machine Communication – Hand-off,
communication, and cooperation between the operator and the autonomous system play an important role to create mutual trust between the human

and the system. It is not known how to address
these issues when the behavior is not known at

design time.


With these enduring challenges in mind, **[P1]**
calls for research to pursue five goals in future technology development. First, approaches to _cumula-_
_tively build safety evidence_ through the phases of
Research & Development (R&D), Test & Evaluation
(T&E), and Operational Tests. The US Air Force
calls for effective methods to reuse safety evidence
throughout the entire product development lifecycle.
Second, **[P1]** argues that _formal methods_, embedded
during R&D, could provide safety assurance. This
approach could reduce the need for T&E and operational tests. Third, novel techniques to _specify re-_
_quirements based on formalism, mathematics, and_
_rigorous natural language_ could bring clarity and
allow automatic test case generation and automated
traceability to low-level designs. Fourth, _run-time_
_decision assurance_ may allow restraining the behavior of the system, thus shifting focus from off-line
verification to instead performing on-line testing at
run-time. Fifth, **[P1]** calls for research on _compo-_
_sitional case generation_, i.e., better approaches to
combine different pieces of evidence into one compelling safety case.


_4.1.2._ _Non-snowballed related work_


This subsection reports the related work that stirred
up the most interesting discussions in the SMILE
project. In contrast to the snowballing literature review, we do not provide steps to replicate the identification of the following papers.
Knauss _et al._ conducted an exploratory interview study to elicit challenges when engineering autonomous cars [51] . Based on interviews and focus


_M. Borg et al. / Safely Entering the Deep_



groups with 26 domain experts in five countries, the
authors report in particular challenges in testing automated vehicles. Major challenges are related to:
1) virtual testing and simulation, 2) safety, reliability, and quality, 3) sensors and their models 4) complexity of, and amount of, test cases, and 5) hand-off
between driver and vehicle.


Spanfelner _et al._ conducted research on safety
and autonomy in the ISO 26262 context [93] . Their
conclusion is that driver assistance systems need
models to be able to interpret the surrounding environment, i.e., to enable vehicular perception. Since
models, by definition, are simplifications of the real
world, they will be subject to functional insufficiencies. By accepting that such insufficiencies may fail
to reach the functional safety goals, it is possible to
design additional measures that in turn can meet the
safety goals.


Heckemann _et al._ identified two primary challenges in developing autonomous vehicles adhering
to ISO 26262 [37] . First, the driver is today considered
to be part of the safety concept, but future vehicles
will make driving maneuvers without interventions
by a human driver. Second, the system complexity
of modern vehicle systems is continuously growing
as new functionality is added. This obstructs safety
assessment, as increased complexity makes it harder
to verify freedom of faults.


Varshney discussed concepts related to engineering safety for ML systems from the perspective of
minimizing risk and epistemic uncertainty [101], i.e.,
uncertainty due to gaps in knowledge as opposed to
intrinsic variability in the products. More specifically, he analyzed how four general strategies for
promoting safety [64] apply to systems with ML components. First, _inherently safe design_ means excluding a potential hazard from the system instead of
controlling it. A prerequisite for assuring such a
design is to improve the interpretability of the typically opaque ML models. Second, _safety reserves_
means the factor of safety, e.g., the ratio of absolute
structural capacity to actual applied load in structural engineering. In ML, interpretations include a
focus on a the maximum error of classifiers instead
of the average error, or training models to be robust
to adversarial examples. Third, _safe fail_ implies that



a system remains safe even when it fails in its intended operation, traditionally by relying on constructs such as electrical fuses and safety valves. In
ML, a concept of run-time monitoring must be accomplished, e.g., by continuously monitoring how
certain a DNN model is performing in its classification task. Fourth, _procedural safeguards_ covers any
safety measures that are not designed into the system, e.g., mandatory safety audits, training of personnel, and user manuals describing how to define
the training set.


Seshia _et al._ identified five major challenges
to achieve formally-verified AI-based systems [86] .
First, a methodology to provide a model of the environment even in the presence of uncertainty. Second, a precise mathematical formulation of what the
system is supposed to do, i.e., a formal specification.
Third, the need to come up with new techniques to
formally model the different components that will
use machine learning. Fourth, systematically generating training and testing data for ML-based components. Finally, developing computationally scalable
engines that are able to verify quantitatively the requirements of a system.


One approach to tackle the opaqueness of DNNs
is to use visualization. Bojarski _et al._ [13] developed a
tool for visualizing the parts of an image that are
used for decision making in vehicular perception.
Their tool demonstrated an end-to-end driving application where the input is images and the output
is the steering angle. Mhamdi _et al._ also studied
the black box aspects of neural networks, and show
that the robustness of a complete DNN can be assessed by an analysis focused on individual neurons
as units of failure [62] – a much more reasonable approach given the state-space explosion.


In a paper on ensemble learning, Varshney _et al._
describes a reject option for classifiers [102] . Such a
classifier could, instead of presenting a highly uncertain classification, request that a human operator
must intervene. A common assumption is that the
classifier is the least confident in the vicinity of the
decision boundary, i.e., that there is an inverse relationship between distance and confidence. While
this might be true in some parts of the feature space,
it is not a reliable measure in parts that contain too


_M. Borg et al. / Safely Entering the Deep_



few training examples. For a reject option to provide
a “safe fail” strategy, it must trigger both 1) near the
decision boundary in parts of the feature space with
many training examples, and 2) in any decision represented by too few training examples.
Heckemann _et al._ proposed using the concept of
_adaptive safety cage architectures_ to support future
autonomy in the automotive domain [37], i.e., an independent safety mechanism that continuously monitors sensor input. The authors separated two areas
of operation: a valid area (that is considered safe)
and an invalid area that can lead to hazardous situ
ations. If the function is about to enter the invalid

area, the safety cage will invoke an appropriate safe
action, such as a minimum risk emergency stopping
maneuver or a graceful degradation. Heckemann
_et al._ argued that a safety cage can be used in an
ASIL decomposition by acting as a functionally redundant system to the actual control system. The
highly complex control function could then be developed according to the quality management standard, whereas the comparably simple safety cage
could adhere to a higher ASIL level.
Adler _et al._ presented a similar run-time monitoring mechanism for detecting malfunctions, referred to as a _safety supervisor_ [1] . Their safety supervisor is part of an overall safety approach for autonomous vehicles, consisting of a structured fourstep method to identify the most critical combinations of behaviors and situations. Once the criti
cal combinations have been specified, the authors
propose implementing tailored safety supervisors to
safeguard against related malfunctions.
Finally, a technical report prepared by Bhattacharyya _et al._ for the NASA Langley Research Center discussed certification considerations
of adaptive systems in the aerospace domain [10] . The
report separates adaptive control algorithms and Artificial Intelligence (AI) algorithms, and the latter is
closely related to our study since it covers machine
learning and ANN. Their certification challenges for
adaptive systems are organized in four categories:


_•_ Comprehensive requirements – Specifying a set
of requirements that completely describe the behavior, as mandated by current safety standards, is
presented as the most difficult challenge to tackle.




_•_ Verifiable requirements – Specifying pass criteria
for test cases at design-time might be hard. Also,
current aerospace V&V relies heavily on coverage
testing of source code in imperative languages, but
how to interpret that for AI algorithms is unclear.


_•_ Documented design – Certification requires detailed documentation, but components realizing
adaptive algorithms were rarely developed with
this in mind. Especially AI algorithms are often
distributively developed by open source communities, which makes it hard to reverse engineer
documentation and traceability.


_•_ Transparent design – Regulators expect a transparent design and a conventional implementation
to be presented for evaluation. Increasing system complexity by introducing novel adaptive algorithms challenges comprehensibility and trust.
On top of that, adaptive systems are often nondeterministic, which makes it harder to demonstrate absence of unintended functionality.


_**4.2.**_ _**The Workshop Series**_


During the six workshops with industry partners (cf.
#1-#6 in Fig. 1), we discussed key questions that
must be explored to enable engineering of safetycritical automotive systems with DNNs. Three subareas emerged during the workshops: 1) robustness,
2) interplay between DNN components and conventional software, and 3) V&V of DNN components.


_4.2.1. Robustness of DNN Components_


The concept of robustness permeated most discussions during the workshops. While robustness is
technically well-defined, in the workshops it often
remained a rather elusive quality attribute – typically
translated to “something you can trust”.
To bring the workshop participants to the same
page, we found it useful to base the discussions on
a simple ML case: a confusion matrix for a oneclass classifier for camera-based animal detection.
For each input image, the result of the classifier is
limited to one of the four options: 1) an animal is
present and correctly classified (true positive), 2) no
animal is present and the classifier does not signal


_M. Borg et al. / Safely Entering the Deep_



animal detection (true negative), 3) the classifier reports animal presence, but there is none (false positive), and 4) an animal is present, but the classifier
misses it (false negative).
For the classifier to be considered robust, the
participants stressed the importance of not generating false positives and false negatives despite occasional low quality input or changes in the environmental conditions, e.g., dusk, rain, or sun glare.
A robust ML system should neither miss present
animals, risking collisions, nor suggest emergency
braking that risk rear-end collisions. As the importance of robustness in the example is obvious, we
see a need for future research both on how to specify and verify acceptable levels of ML robustness.
During the workshops, we also discussed
more technical aspects engineering robust DNN
components. First, our industry practitioners
brought up the issue of DNN architectures to
be problem-specific. While there are some approaches to automatically generating neural network
architectures [6] _[,]_ [104], typically designing the DNN architecture is an _ad hoc_ process of trial and error.
Often a well known architecture is used as a base
line and then it is tuned to fit the problem at hand.
Our workshops recognized the challenge of engineering robust DNN-based systems, in part due to
their highly problem-specific architectures.

Second, once the DNN architecture is set, training commences to assign weights to the trainable
parameters of the network. The selection of training data must be representative for the task, in our
discussions animal detection, and for the environment that the system will operate in. The workshops agreed that robustness of DNN components
can never be achieved without careful selection of

training data. Not only must the amount and quality
of sensors (in our case cameras) acquiring the different stimuli for the training data be sufficient, also
other factors such as positioning, orientation, aperture, and even geographical location like city and
country must match the animal detection example.
At the workshops, we emphasized the issue of cam

_§_
http://torcs.sourceforge.net

_¶_
http://www.cvlibs.net/datasets/kitti/
_∥_ https://www.cityscapes-dataset.com/



era positions as both car and truck manufacturers
were part of SMILE – to what extent can training
data from a car’s perspective be reused for a truck?
Or should a truck rather benefit from its size and collect dedicated training data from its elevated camera
position?
Third, also related to training data, the workshops discussed working with synthetic data. While
such data always can be used to complement training data, there are several open questions on how to
best come up with the best mix during the training
stage. As reported in Section 2.2, GANs [30] _[,]_ [74] could
be a good tool for synthesizing data. Sixt _et al._ [90]

proposed a framework called RenderGAN that could
generate large amounts of realistic labeled data for
training. In transfer learning, training efficiency improves by combining data from different data sets
29 _,_ 69 . One possible approach could be to first train
the DNN component using synthetic data from, e.g.,
simulators like TORCS [§], then data from some publicly available database could be used to continue
the training, e.g., the KITTI data [¶] or CityScape, and _[∥]_
finally, data from the geographical region where the
vehicle should operate could be added. For any attempts at transfer learning, the workshops identified the need to measure to what extent training data
matches the planned operational environment.


_4.2.2._ _Complementing DNNs with Conventional_

_Components_


During the workshops, we repeatedly reminded the
participants to consider DNNs from a systems perspective. DNN components will always be part of an
automotive system consisting of also conventional
hardware and software components.
Several researchers claim that that DNN components is a prerequisite for autonomous driving [28] _[,]_ [2] _[,]_ [79] .
However, how to integrate such components in a
system is an open question. Safety is a systems
issue, rather than a component specific issue. All
hazards introduced by both DNNs and conventional
software must be analyzed within the context of sys

_M. Borg et al. / Safely Entering the Deep_



tems engineering principles. On the other hand, the
hazards can also be addressed on a system level.


One approach to achieve DNN safety is to introduce complementary components, i.e., when a DNN
model fails to generalize, a conventional software
or hardware component might step in to maintain
safe operation. During the workshops, particular attention was given to introducing a _safety cage con-_
_cept_ . Our discussions orbited a solution in which
the DNN component was encapsulated by a supervisor, or a safety cage, that continuously monitors
the input to the DNN component. The envisioned
safety cage should perform novelty detection [70] and
alert when input does not belong within the training region of the DNN component, i.e., if the risk of
failed generalization was too high, the safety cage
should re-direct the execution to a _safe-track_ . The
safe-track should then operate without any ML components involved, enabling traditional approaches to
safety-critical software engineering.


The concept of an ML safety cage is in line
with Varshney’s discussions of “safe fail” [101] . Different options to implement an ML safety cage include adaptations of fail-silent systems [17], plausibility checks [52], and arbitration. However, Adler _et al._ [1]

indicated that the _no free lunch_ theorem might apply for safety cages, by stating that if tailored safety
safety cages are to be developed to safeguard against
domain-specific malfunctions, thus, different safety
cages may be required for different systems.


Introducing redundancy in the ML system is an
approach related to the safe track. One method
is to use ensemble methods in computer vision
applications [61], i.e., employing multiple learning algorithms to improve predictive performance. Redundancy can also be introduced in an ML-based
system using hardware component, e.g., using an array of sensors of the same, or different, kind. Increasing the amount of input data should increase
the probability of finding patterns closer to the training data set. Combining data from various input
sources, referred to as sensor fusion, also helps overcoming the potential deficiencies of individual sen
sors.



_4.2.3._ _V&V Approaches for Systems with DNN_

_Components_


Developing approaches to engineer robust systems
with DNN components is not enough, the automotive industry must also develop novel approaches to
V&V. V&V is a cornerstone in safety certification,
but it still remains unclear how to develop a safety
case around applications with DNNs.
As pointed out in previous work, the current
ISO 26262 standard is not applicable when developing autonomous systems that rely on DNNs [37] . Our
workshops corroborate this view, by identifying several open questions that need to be better understood:


_•_ How is a DNN component classified in
ISO 26262? Should it be regarded as an individual software unit or a component?


_•_ From a safety perspective, is it possible to treat
DNN misclassifications as “hardware failures”? If
yes, are the hardware failure target values defined
in ISO 26262 applicable?


_•_ ISO 26262 mandates complete test coverage of
the software, but what does this imply for a DNN?
What is sufficient coverage for a DNN?


_•_ What metrics should be used to specify the DNN
accuracy? Should quality targets using such metrics be used in the DNN requirements specifications, and subsequently as targets for verification
activities?


Apart from the open questions, our workshop
participants identified several aspects that would
support V&V. First, as requirements engineering is
fundamental to high-quality V&V [12], some workshop participants requested a formal, or semiformal, notation for requirements related to functional safety in the DNN context. Defining lowlevel requirements that would be verifiable appears
to be one of the greatest challenges in this area.
Second, there is a need for a tool-chain and framework tailored to lifecycle management of systems
with DNN components – current solutions tailored
for human-readable source code are not feasible and

must be complemented with too many immature internal tools. Third, methods for test case generation


_M. Borg et al. / Safely Entering the Deep_



for DNN will be critical, as manual creation of test

data does not scale.

Finally, a major theme during the workshops was
how to best use simulation as a means to support
V&V. We believe that the future will require massive use of simulation to ensure safe DNN com
ponents. Consequently, there is a need to develop
simulation strategies to cover both normal circumstances as well as rare, but dangerous, traffic situations. Furthermore, simulation might also be used to
assess the sensitivity to adversarial examples.


_**4.3.**_ _**The systematic snowballing**_


Table 5 shows the results from the five iterations
of the snowballing. In total, the snowballing procedure identified 64 papers including the start set.
We notice two publication peaks: 29 papers were
published between 2002-2007 and 25 papers were
published between 2013-2016. The former set of
papers were dominated by research on using neural networks for adaptive flight controllers, whereas
the latter set predominantly addresses the automotive domain. This finding suggests that organizations currently developing ML-based systems for
self-driving cars could learn from similar endeavors in the aerospace domain roughly a decade ago
– while DNN was not available then, several aspects
of V&V enforced by aerospace safety standards are
similar to ISO 26262. Note, however, that 19 of the
papers do not target any specific domain, but rather
discusses ML-based systems in general.
Table 2 shows the distribution of challenge and
solution proposal categories identified in the papers;
‘#’ indicates the number of unique challenges or solution proposals matching a specific category. As
each paper can report more than one challenge or
solution proposal, and the same challenge or solution proposal can occur in more than one paper, the
number of Paper IDs in the third column does not
necessarily match the ‘#’. The challenges most frequently mentioned in the papers relate to state-space
explosion and robustness, whereas the most commonly proposed solutions constitute approaches that
belong to formal methods, control theory, or proba


bilistic methods.

Regarding the publication years, we notice that
the discussion on state-space explosion primarily
has been active in recent years, possibly explained
by the increasing application of DNNs. Looking at
solution proposals, we see that probabilistic methods was particularly popular during the first publication peak, and that research specifically addressing
test case design for ML-based systems has appeared
first after 2012.
Fig. 3 shows a mapping between solution proposals categories and challenge categories. Some of
the papers propose a solution to address challenges
belonging to a specific category. For each such instance, we connect solution proposals (to the left)
and challenges (to the right), i.e., the width of the
connection illustrates the number of instances. Note

that we did put the solution proposal in **[P4]** (deployment in real operational setting) in its own ‘Other’
category. None of the proposed solutions address
challenges related to the categories “Requirements
specification” or “Systems engineering”, indicating
a research gap. Furthermore, “Transparency” is the
challenge category that has been addressed the most
in the papers, followed by “State-space explosion”.


Fig. 3. Mapping between categories of solution proposals
(to the left) and challenges (to the right).


Two books summarize most findings from the
aerospace domain identified through our systematic
snowballing. Taylor edited a book in 2006 that collected experiences for V&V of ANN technology [96]

in a project sponsored by the NASA Goddard Space
Flight Center. Taylor concluded that the V&V techniques available at the time must evolve to tackle



_∗∗_ The best practices were also later distilled into a guidance document intended for practitioners 73


_M. Borg et al. / Safely Entering the Deep_


Table 2. Distribution of challenge and solution proposal categories.


|Challenge category|#|Paper IDs|
|---|---|---|
|State-space explosion|6|[P3], [P15], [P16], [P47]|
|Robustness|4|[P1], [P2], [P15], [P55]|
|Systems engineering|2|[P1], [P55]|
|Transparency|2|[P1], P55]|
|Requirements speciﬁcation|3|[P15], [P55]|
|Test speciﬁcation|3|[P16], [P46], [P55]|
|Adversarial attacks|1|[P15]|


|Solution proposal category|#|Paper IDs|
|---|---|---|
|Formal methods|8|[P3], [P26], [P42], [P28],<br>[P37], P[40], [P44], [P53]|
|Control theory|7|[P7], [P20], [P25], [P64],<br>[P36], [P47], [P57], [P60]|
|Probabilistic methods|7|[P18], [P30], [P31], [P32], [P33],<br>[P35], [P50], [P52], [P54]|
|Test case design|5|[P9], [P10], [P12], [P17], [P21]|
|Process guidelines|4|[P23], [P51], [P56], [P59]|



ANNs. Taylor’s book reports five areas that need
to be augmented to allow V&V of ANN-based systems: _[∗∗]_


_•_ _Configuration management_ must track all additional design elements, e.g., the training data, the
network architecture, and the learning algorithms.
Any V&V activity must carefully specify the configuration under test.


_•_ _Requirements_ need to specify novel adaptive behavior, including control requirements (how to acquire and act on knowledge) and knowledge requirements (what knowledge should be acquired).


_•_ _Design specifications_ must capture design choices
related to novel design elements such as training
data, network architecture, and activation functions. V&V of the ANN design should ensure that
the choices are appropriate.


_•_ _Development lifecycles_ for ANNs are highly iterative and last until some quantitative goal has been
reached. Traditional waterfall software development is not feasible, and V&V must be an integral
part rather than an add-on.


_•_ _Testing_ needs to evolve to address novel re


quirements. Structure testing should determine
whether the network architecture is better at learn
ing according to the control requirements than alternative architectures. Knowledge testing should
verify that the ANN has learned what was specified in the knowledge requirements.


The second book that has collected experiences
on V&V of (mostly aerospace) ANNs, also funded
by NASA, was edited by Schumann and Liu and
published in 2010 [83] . While the book primarily surveys the use of ANNs in high-assurance systems,
parts of the discussion is focused on V&V – and the
overall conclusion that V&V must evolve to handle

ANNs is corroborated. In contrast to the organization we report in Table 2, the book suggests grouping
solution proposals into approaches that: 1) separate
ANN algorithms from conventional source code, 2)
analyze the network architecture, 3) consider ANNs
as function approximators, 4) tackle the opaqueness
of ANNs, 5) assess the characteristics of the learning algorithm, 6) analyze the selection and quality of
training data, and 7) provides means for online monitoring of ANNs. We believe that our organization is


_M. Borg et al. / Safely Entering the Deep_



largely orthogonal to the list above, thus both could
be used in a complementary fashion.


_**4.4.**_ _**The survey**_


This section organizes the findings from the survey
into closed questions, correlation analysis, and open
questions, respectively.


_4.4.1._ _Closed questions_


Forty-nine practitioners answered our survey, most
of them primarily working in Europe (38 out of 49,
77.6%). Twenty respondents (40.8%) work primarily in the automotive domain, followed by 14 in
aerospace (28.6%). Other represented domains include process industry (5 respondents), railway (5
respondents), and government/military (3 respondents). The respondents represent a variety of roles,
from system architects (17 out of 49, 34.7%) to
product developers (10 out of 49, 20.4%), and managerial roles (7 out of 49, 14.3%). Most respondents
primarily work in Europe (38 out of 49, 77.6%) or
North America (7 out of 49, 14.3%).
Most respondents have some proficiency in ML.
Twenty-five respondents (51.0%) report having fundamental awareness of ML concepts and practical
ML concerns. Sixteen respondents (32.7%) have
higher proficiency, i.e., can implement ML solutions
independently or with guidance – but no respondents
consider themselves ML experts. On the other side
of the spectrum, eight respondents report possessing
no ML knowledge.
We used three Likert items to assess the respondents’ general thoughts about ML and functional
safety, reported as a)-c) in Table 3. Most respondents agree (or strongly agree) that applying ML in
safety-critical applications will be important in their
organizations in the future (29 out of 49, 59.2%),
whereas eight (16.3%) disagree. At the same time,
29 out of 49 (59.2%) of the respondents report that
V&V of ML-based features is considered particularly difficult by their organizations – 20 respondents
even strongly agrees with the statement. It is clear
to our respondents that more attention is needed regarding V&V of ML-based systems, as only 10 out



of 49 (20.4%) believe that their organizations are
well-prepared for the emerging paradigm.
Robustness (cf. e) in Table 3) stands out as the
particularly important challenge, reported as “extremely important” by 29 out of 49 (59.2%). However, all challenges covered in the questionnaire
were considered important by the respondents. The
only challenge that appears less urgent to the respondents is adversarial attacks, but the difference is mi
nor.

The respondents consider simulated test cases as
the most promising solution proposal to tackle challenges in V&V of ML-based systems, reported as
extremely promising by 18 out of 49 respondents
(36.7%) and moderately promising by 12 respondents (24.5%). Probabilistic methods is the least
promising solution proposal according to the respondents, followed by process guidelines.


_4.4.2._ _Correlation analysis_


We identified some noteworthy correlations in the
responses. The respondents’ ML proficiency (Q4)
is moderately correlated ( _ρ_ = 0 _._ 53) with the perception of ML importance (Q5) – an expected finding as respondents with a personal investment are
likely to be biased. More interestingly, we found that
ML proficiency was also moderately correlated to
two of the seven challenge categories: transparency
( _ρ_ = 0 _._ 61) and state-space explosion ( _ρ_ = 0 _._ 54).
This suggests that these two challenges are particularly difficult to comprehend for non-experts. Perceiving the organization as well-prepared for introducing ML-based solutions (Q4) is moderately correlated ( _ρ_ = 0 _._ 57) with considering systems engineering challenges (Q7) as particularly important
and weakly correlated regarding process guidelines
(Q16) as a promising solution ( _ρ_ = 0 _._ 37). As these
are the only correlations with Q4, it indicates that organizations that have reached a certain ML maturity
have progressed beyond specific issues and instead
focus on the bigger picture, i.e, how to incorporate
ML in systems and how to adapt internal processes
in the new ML era.

There are more correlations within the cate
gories of challenges (Q5-Q11) and solution proposals (Q12-Q16) than between the two groups. The


_M. Borg et al. / Safely Entering the Deep_


Table 3. Answers to the closed questions of the survey. a)-c)
show three Likert items, ranging from strongly disagree (1) to
strongly agree (5). d)- o) reports on importance/promisingness
using the following ordinal scale: not at all, slightly, somewhat,
moderately, and extremely. The ‘Missing‘” column includes
both “I don’t know” answers and missing answers.


_M. Borg et al. / Safely Entering the Deep_



only strong correlation between groups is test specification (Q11) and formal methods (Q12) ( _ρ_ =
0 _._ 71). Within the challenges, the correlation between the two challenges state-space explosion (Q5)
and transparency (Q8) stands out as particularly
strong ( _ρ_ = 0 _._ 91), illustrating the close connection between these two issues with large DNN architectures. Also the two challenge categories requirements specifications (Q9) and test specifications (Q11) are strongly correlated ( _ρ_ = 0 _._ 71), in
line with a large body of previous work on aligning
the two concepts [12] .


_4.4.3._ _Open questions_


The end of the questionnaire contained an openended question (Q17), requesting a comment on
Fig. 3 and the accompanying findings: “although
few individual V&V challenges related to machine
learning transparency are highlighted in the literature, it is the challenge most often addressed by the
previous publications’ solution proposals. We also
find that the second most addressed challenge in previous work is related to state-space explosion.”
Sixteen out of 49 respondents (32.7%) provided
a free text answer to Q17, representing highly contrasting viewpoints. Eight respondents reported that
the findings were not in line with their expectations, whether seven respondents agreed – one respondent neither agreed nor disagreed. Examples
of more important challenges emphasized by the respondents include both other listed challenges, i.e.,
robustness and requirements specification, and other
challenges, e.g., uncertainty of sensor data (in automotive) and the knowledge gap between industry and regulatory bodies (in the process industry).
Three respondents answer in general terms that the
main challenge of ML-based systems is the intrinsic
non-determinism.

On the other hand, the agreeing respondents motivate that state-space explosion is indeed the most
pressing challenge due to the huge input space of
the operational environment (both in automotive and
railway applications). One automotive researcher
stresses that the state-space explosion impedes rigid
testing but raises the transparency challenge as well
– a lack thereof greatly limits analyzability, which is



a key requirement for safety-critical systems. One
automotive developer argues that the bigger statespace of the input domain, the bigger the attack surface becomes – possibly referring to both adversarial attacks and other antagonistic cyber attacks. Finally, two respondents provide answers that encourage us to continue work along to paths in the SMILE
project: 1) a tester in the railway domain explains
that the traceability during root cause analyses in
ML-applications will be critical, in line with our
argumentation at a recent traceability conference [15],
and 2) one automotive architect argues that the
state-space explosion will not be the main challenge
as any autonomous driving will have to be within
“guard rails”, i.e., a solution similar to the safety
cage architectures we intend to develop in the next
phase of the project.
Seven respondents complemented the survey answers with concluding thoughts in Q18. One experienced manager in the aerospace domain explained:
“What is now called ML was called neural nets (but
less sophisticated) 30 years ago.”, a statement that
supports our recommendation that the automotive
industry should aim for a cross-domain knowledge
transfer regarding V&V of ML-based systems. The
manager followed by stating: “it (ML) introduces
a new element in safety engineering. Or at least it
moves the emphasis to more resilience. If the classifier is wrong, then it becomes a hazard and the system must be prepared for it.” We agree with the respondent that actions needed in the hazardous situation must be well-specified. Two respondents comment that conservatism is fundamental in functional

safety, one of them elaborates that the “end of predictability” introduced by ML is a disruptive change
that requires a paradigm shift.


**5.** **Revisiting the RQs**


This section first discusses the RQs in a larger context, and then aggregates the four sources of evidence presented in Fig. 2. Finally, we discuss implications for research and practice, including automotive manufacturers and regulatory bodies, and
conclude by reporting the main threats to validity.
Table 4 summarizes our findings.


_M. Borg et al. / Safely Entering the Deep_



_**5.1.**_ _**RQ1: State-of-the-art in V&V of**_
_**safety-critical ML**_


There is no doubt that deep learning research currently has incredible momentum. New applications
and success stories are reported every month – and
many applications come from the automotive domain. The rapid movement of the field is reflected
by the many papers our study has identified on
preprint archives, in particular the arXiv.org e-Print
archive. It is evident that researchers are eager to
claim novelty, and thus struggle to publish results as
fast as possible.
While DNNs have enabled amazing breakthroughs, there is much less published work on engineering safety for DNNs. On the other hand, we
observe a growing interest as several researchers call
for more research on DNN safety, as well as ML
safety in general. However, there is no agreement
on how to best develop safety-critical DNNs, and
several different approaches have been proposed.
Contemporary research endeavors often address the
opaqueness of DNNs, to support analyzability and
interpretability of systems with DNN components.
Deep learning research is in its infancy, and the
tangible pioneering spirit sometimes brings the mind
to the Wild West. Anything goes, and there is a potential for great academic recognition for groundbreaking papers. There is certainly more fame in
showcasing impressive applications than updating
engineering practices and processes.
Safety engineering stands as a stark contrast to
the pioneering spirit. On the contrary, safety is permeated by conservatism. When a safety standard
is developed, it captures the best available practices to engineer safe systems. This approach inevitably results in standards that lag behind the research front – safety first! In the automotive domain,
ISO 26262 was developed long before DNNs for vehicles was an issue. Without question, DNNs constitute a paradigm shift in how to approach functional
safety certification for automotive software, and we
do not believe in any quick fixes to patch ISO 26262
for this new era. As recognized by researchers before us, e.g., Salay _et al._ [78], there is a considerable
gap between ML and ISO 26262 – a gap that probably needs to be bridged by new standards rather than



incremental updates of previous work.
Broadening the discussion from DNNs to ML in
general, our systematic snowballing of previous research on safety-critical shows a peak of aerospace
research between 2002-2007 and automotive re
search dominating from 2013 and onwards. We notice that the aerospace domain allocated significant
resources to research on neural networks for adaptive flight controllers roughly a decade before DNNs
became popular in automotive research. We hypothesize that considerable knowledge transfer between
the domains is possible now, and plan to proceed
such work in the near future.

The academic literature on challenges in MLbased safety engineering has most frequently addressed state-space explosion and robustness (see
Table 1 for definitions). On the other hand, the
most commonly proposed solutions to overcome
challenges of ML-based safety engineering are approaches that belong to formal methods, control theory, or probabilistic methods – but these appear to be
only moderately promising by industry practitioners, who would rather see research on simulated test
cases. As discussed in relation to RQ2, academia
and industry share a common view on what challenges are important, but the level of agreement on
what is the best way forward appears to be less clear.


_**5.2.**_ _**RQ2: Main challenges for safe automotive**_

_**DNNs**_


Industry practice is far from certifying DNNs for
use in driverless safety-critical applications on public roads. Both the workshop series and the survey show that industry practitioners across organizations do not know how to tackle the challenge of approaching regulatory bodies and certification agencies with DNN-based systems. Most likely, both automotive manufacturers and safety standards need to
largely adapt to fit the new ML paradigm – the current gap appears not to be bridgeable in the foreseeable future through incremental science alone.
On the other hand, although the current safety
standards do not encompass ML yet, several automotive manufacturers are highly active in engineering autonomous vehicles. Tesla has received significant media coverage through pioneering demonstra

_M. Borg et al. / Safely Entering the Deep_



tions and self-confident statements. Volvo Cars is
also highly active through the Drive Me initiative,
and has announced a long-lasting partnership with
Uber toward autonomous taxis.


Several other partnerships have recently been
announced among automotive manufacturers, chipmakers, and ML-intensive companies. For example, Nvidia has partnered with Uber, Volkswagen,
and Audi to support engineering self-driving cars
using their GPU computing technology for ML development. Nvidia has also partnered with the Internet company Baidu, a company that has a highly
competitive ML research group. Similarly, the chipmaker Intel has partnered with Fiat Chrysler Automobiles and the BMW Group to develop autonomy
around their Mobileye solution. Moreover, large
players such as Google, Apple, Ford, and Bosch are
active in the area, as well as startups such as nuTonomy and FiveAI – no one wants to miss the boat to
the lucrative future.


While there are impressive achievements both
from spearheading research, and some features are
already available on the consumer market, they all
have in common that the safety case argumentation
relies on a human-in-the-loop. In case there is a
critical situation, the human driver is expected to
be present and take control over the vehicle. There
are joint initiatives to formulate regulations for autonomous vehicles, but, analogously, there is a need
for initiatives paving the way for new standards addressing functional safety of systems that rely on
ML and DNNs.


We elicited the most pressing issues concerning
engineering of DNN-based systems through a workshop series and a survey with practitioners. Many
discussions during the workshops were dominated
by robustness of DNN components, including detailed considerations about robust DNN architec
tures and the requirements on training data to learn
a robust DNN model. Also the survey shows the
importance of ML robustness, which motivates the
attention it has received in academic publications
(cf. RQ1). On the other hand, while there is an
agreement on the importance of ML robustness between academia and industry, how to tackle the phenomenon is still an open question – and thus a po


tential avenue for future research. Nonetheless, the
problem of training a robust DNN component corresponding to the complexity of public traffic conforms with several of the “enduring problems” highlighted by the US Air Force in their technical report
on V&V of autonomous systems **[P1]**, e.g., statespace explosion and unpredictable environments.
While robustness is stressed by practitioners,
academic publications have instead to a larger extent
highlighted challenges related to the limited transparency of ML-based systems (e.g., Bhattacharyya
_et al._ [10] ) and the inevitable state-space explosion.
The survey respondents confirm these challenges
as well, but we recommend future studies to meet
the expectations from industry regarding robustness
research. Note, however, that the concept of robustness might have different interpretations despite
having a formal IEEE definition [42] . Consequently,
we call for an empirical study to capture what industry means by ML and DNN robustness in the automotive context.

The workshop participants perceived two possible approaches to pave the way for safety-critical
DNNs as especially promising. First, continuous
monitoring of DNN input using a safety cage architecture, a concept that has been proposed for
example by Adler _et al._ [1] . Monitoring safe operation, and re-directing execution to a “safe track”
without DNN involvement when uncertainties grow
too large, is an example of the safety strategy safe
fail [101] . Another approach to engineering ML safety,
considered promising by the workshops and the survey respondents alike, is to simulate test cases.


**6.** **Conclusion and future work**


Deep learning Neural Networks (DNN) is key to
enable the vehicular perception required for autonomous driving. However, the behavior of DNN
components cannot be guaranteed by traditional
software and system engineering approaches. On
top of that, crucial parts of the automotive safety
standard ISO 26262 are not well-defined for certifying autonomous systems [78] _[,]_ [39] – certain process requirements contravene the nature of developing Machine Learning (ML)-based systems, especially in


_M. Borg et al. / Safely Entering the Deep_


Table 4: Condensed findings in relation to the research questions, and implications for research and practice.








|RQ1. What is the state-of-the-<br>art in V&V of ML-based safety-<br>critical systems?|• Most ML research showcases applications, while development on ML V&V is<br>lagging behind.<br>• Considerable gap between V&V mandated by safety standards and nature of<br>contemporary ML-based systems.<br>• The aerospace domain has collected experiences from V&V of adaptive flight<br>controllers based on neural networks.<br>• Support for V&V of ML-based systems can be organized into: 1) formal meth-<br>ods, 2) control theory, 3) probabilistic methods, 4) process guidelines, and 5)<br>simulated test cases.<br>• Academia has focused mostly on 1)–3), whereas industry perceives 5) as the<br>most promising.|
|---|---|
|RQ2. What are the main chal-<br>lenges when engineering safety-<br>critical systems with DNN com-<br>ponents in the automotive do-<br>main?|_•_ How to certify safety-critical systems with DNNs for use on public roads is<br>unclear.<br>_•_ Industry stresses robustness, whereas academia most often addresses state-<br>space explosion and the lack of ML transparency.<br>_•_ Challenges elicited corroborate work on V&V by NASA and USAF, covering<br>neural networks, autonomous systems, and adaptive systems.|
|Implications for research and<br>practice|_•_ Gap between ML practice and ISO 26262 requires novel standards rather than<br>incremental updates.<br>_•_ Cross-domain knowledge transfer from the aerospace V&V engineers to the<br>automotive domain appears promising.<br>_•_ Need for empirical studies to clarify what robustness means in the context of<br>DNN-based autonomous vehicles.<br>_•_ Systems-based safety approaches encouraged by industry, including safety cage<br>architectures and simulated test cases.|


_M. Borg et al. / Safely Entering the Deep_



relation to Verification and Validation (V&V).


Roughly a decade ago, using Artificial Neural
Networks (ANN) in flight controllers was an active research topic, and also how to adhere to the
strict aerospace safety standards. Now, in the advent of autonomous driving, we recommend the automotive industry to learn from guidelines [73] and
lessons learned [96] from V&V of ANN-based com
ponents developed to conform with the DO-178B
software safety standard for airborne systems. In
particular, automotive software developers need to
evolve practices for _configuration management_ and
_architecture specifications_ to encompass fundamental DNN design elements. Also, _requirements spec-_
_ifications_ and the corresponding _software testing_
must be augmented to address the adaptive behavior of DNNs. Finally, the highly _iterative develop-_
_ment lifecycle of DNNs_ should be aligned with the
traditional automotive V-model for systems development. A recent NASA report on safety certification
of adaptive aerospace systems [10] confirms the challenges of requirements specification and software
testing. Moreover, related to ML, the report adds
the _lack of documentation and traceability_ in many
open source libraries, and the issue of an _exper-_
_tise gap between regulators and engineers_ – conventional source code in C/C++ is very different from an
opaque ML model trained on a massive dataset.


The work most similar to ours also originated in
the aerospace domain, i.e., a project initiated by the
US Air Force to describe enduring problems (and future possibilities) in relation to safety certification of
autonomous systems **[P1]** . The project highlighted
four primary challenges: 1) state-space explosion,
2) unpredictable environments, 3) emergent behavior, and 4) human-machine communication. While
not explicitly discussing ML, the first two findings
match the most pressing needs elicited in our work,
i.e., _state-space explosion as stressed by the aca-_
_demic literature_ (in combination with limited transparency) and _robustness as emphasized by the work-_
_shop participants as well as the survey respondents_
(referred to as unpredictable environments in **[P1]** ).


After having reviewed the state-of-the-art and
state-of-practice, the SMILE project will now embark on a solution-oriented journey. Based on the



workshops, and motivated by the survey respondents, we conclude that _pursuing a solution based_
_on safety cage architectures_ [37] _[,]_ [1] encompassing DNN
components is a promising direction. Our rationale
is three-fold. First, the results from the workshops
with automotive experts from industry clearly motivates us, i.e., the participants strongly encouraged
us to explore such a solution as the next step. Second, we believe it would be feasible to develop a
_safety case_ around a safety cage architecture, since
the automotive industry already uses the concept in
the physical vehicles. Third, we believe the DNN
technology is ready to provide what is needed in
terms of novelty detection. The safety cage architecture we envision will continuously monitor input data from the operational environment to redirect execution to a non-ML safe track when un
certainties grow too large. Consequently, we advocate _DNN safety strategies using a systems-based_
_approach_ rather than techniques that focus on the internals of DNNs. Finally, also motivated by both the
workshops and the survey respondents, we propose
an approach to V&V that makes heavy use of _sim-_
_ulation_ – in line with previous recommendations by
other researchers [9] _[,]_ [14] _[,]_ [99] .


Future work will also study how _transfer learn-_
_ing_ could be used to incorporate training data from
different contexts or manufacturers, or even include
synthetic data from simulators, into DNNs for realworld automotive perception. So far we have mostly
limited the discussion to fixed DNN-based systems,
i.e., systems trained only prior to deployment. An
obvious direction for future work is to to explore
how dynamic DNNs would influence our findings,
i.e., DNNs that adapt by continued learning either in
batches or through online learning. Furthermore, research on V&V of ML-based systems is more complex than pure technology in isolation. Thus, we
recognize the need to explore both ethical and legal aspects involved in safety certification of MLbased systems. Finally, there is a new automotive
standard under development that will address autonomous safety: ISO/PAS 21448 Road vehicles –
Safety of the intended functionality. We are not
aware of its contents at the time of this writing, but
once published, we will use it as an important refer

_M. Borg et al. / Safely Entering the Deep_



ence point for our future solution proposals.


**Acknowledgments**


Thanks go to all participants in the SMILE workshops, in particular Carl Zand´en, Michal Simoen,
and Konstantin Lindstr¨om. This work was carried

out within the SMILE and SMILE II projects financed by Vinnova, FFI, Fordonsstrategisk forskning och innovation under the grant numbers: 201604255 and 2017-03066.


**References**


1. R. Adler, P. Feth, and D. Schneider. Safety Engineering for Autonomous Vehicles. In _Proc. of the 46th An-_
_nual IEEE/IFIP International Conference on Depend-_
_able Systems an dNetworks Workshops_, pages 200–
205, 2016.
2. Ankur Agrawal, Chia-Yu Chen, Jungwook Choi,
Kailash Gopalakrishnan, Jinwook Oh, Sunil Shukla,
Viji Srinivasan, Swagath Venkataramani, and Wei
Zhang. Accelerator Design for Deep Learning Training: Extended Abstract: Invited. In _Proc. of the 54th_
_Annual Design Automation Conference_, pages 57:1–
57:2, 2017.
3. A. K. Akametalu, J. F. Fisac, J. H. Gillula, S. Kaynama, M. N. Zeilinger, and C. J. Tomlin. Reachabilitybased safe learning with Gaussian processes. In _Proc._
_of the 53rd IEEE Conference on Decision and Con-_
_trol_, pages 1424–1431, 2014.
4. Rob Alexander, Heather Rebecca Hawkins, and Andrew John Rae. Situation coverage - A coverage criterion for testing autonomous robots. Technical report,
University of York, 2015.
5. Dario Amodei, Chris Olah, Jacob Steinhardt, Paul
Christiano, John Schulman, and Dan Mane. Concrete
Problems in AI Safety. _arXiv:1606.06565_, 2016.
6. P. J. Angeline, G. M. Saunders, and J. B. Pollack. An
evolutionary algorithm that constructs recurrent neural networks. _IEEE Transactions on Neural Networks_,
5(1):54–65, 1994.
7. James Arnold and Rob Alexander. Testing Autonomous Robot Control Software Using Procedural
Content Generation. In _Computer Safety, Reliability,_
_and Security_, pages 33–44. Springer, Berlin, 2013.
8. Nicholas J. Bahr. _System Safety Engineering and Risk_
_Assessment: A Practical Approach, Second Edition_ .
CRC Press, 2014.
9. Raja Ben Abdessalem, Shiva Nejati, Lionel C Briand,
and Thomas Stifter. Testing advanced driver assistance systems using multi-objective search and neural



networks. In _Proc. of the 31st International Confer-_
_ence on Automated Software Engineering_, pages 63–
74. ACM, 2016.
10. Siddhartha Bhattacharyya, Darren Cofer, D Musliner,
Joseph Mueller, and Eric Engstrom. Certification considerations for adaptive systems. Technical Report
CR2015-218702, NASA, 2015.
11. R. Billinton and R. Allan, editors. _Reliability evalua-_
_tion of engineering systems: concepts and techniques_ .
Springer Science & Business Media, 2013.
12. E. Bjarnason, P. Runeson, M. Borg, M. Unterkalmsteiner, E. Engstrom, B. Regnell, G. Sabaliauskaite,
A. Loconsole, T. Gorschek, and R. Feldt. Challenges
and practices in aligning requirements with verification and validation: a case study of six companies.
_Empirical Software Engineering_, 19(6):1809–1855,
2014.
13. Mariusz Bojarski, A. Choromanska, K. Choromanski, B. Firner, L. Jackel, U. Muller, and K. Zieba.
VisualBackProp: efficient visualization of CNNs.
_arXiv:1611.05418_, 2016.
14. Mariusz Bojarski, Davide Del Testa, Daniel
Dworakowski, Bernhard Firner, Beat Flepp, Prasoon Goyal, Lawrence D. Jackel, Mathew Monfort,
Urs Muller, Jiakai Zhang, Xin Zhang, Jake Zhao, and
Karol Zieba. End to End Learning for Self-Driving
Cars. _arXiv:1604.07316_, 2016. arXiv: 1604.07316.
15. M. Borg, C. Englund, and B. Duran. Traceability and Deep Learning - Safety-critical Systems with
Traces Ending in Deep Neural Networks. In _Proc. of_
_the Grand Challenges of Traceability: The Next Ten_
_Years_, pages 48–49, 2017.
16. J. Bosworth and P. Williams-Hayes. Flight Test Results from the NF-15b Intelligent Flight Control System Project with Adaptation to a Simulated Stabilitor
Failure. Technical report, NASA TM-2007-214629,
2007.
17. F. V. Brasileiro, P. D. Ezhilchelvan, S. K. Shrivastava,
N. A. Speirs, and S. Tao. Implementing fail-silent
nodes for distributed systems. _IEEE Transactions on_
_Computers_, 45(11):1226–1238, 1996.
18. G. Brat and A. Jonsson. Challenges in verification and
validation of autonomous systems for space exploration. In _Proc. of the IEEE International Joint Con-_
_ference on Neural Networks_, volume 5, pages 2909–
2914 vol. 5, 2005.
19. R. L. Broderick. Statistical and adaptive approach for
verification of a neural-based flight control system. In
_Proc. of the 23rd Digital Avionics Systems Confer-_
_ence_, volume 2, pages 6.E.1–61–10, 2004.
20. R. L. Broderick. Adaptive verification for an on-line
learning neural-based flight control system. In _Proc._
_of the 24th Digital Avionics Systems Conference_, volume 1, pages 6.C.2–61–10, 2005.
21. A. Broggi, M. Buzzoni, S. Debattisti, P. Grisleri,


_M. Borg et al. / Safely Entering the Deep_


Table 5: The start set and the four subsequent iterations of the snowballing literature review.


|Start set|[P1] M. Clark et al.24, [P2] D. Amodei et al.5, [P3] G. Brat and A. Jonsson18, [P4] A. Broggi et<br>al.21, [P5] B. Taylor et al.97, [P6] J. Taylor et al.98, [P7] A. Carvalho et al.23, [P8] S. Ramos et<br>al.75, [P9] R. Alexander et al.4, [P10] X. Zou et al.112, [P11] X. Zou et al.113, [P12] J. Arnold and<br>R. Alexander7, [P13] S. Sivaraman and M. Trivedi89, [P14] A. Mozaffari et al.65|
|---|---|
|Iteration 1|**[P15]** S. Seshia_ et al._ ~~86~~,** [P16]** P. Helle_ et al._~~38~~,** [P17]** L. Li_ et al._,~~57~~,** [P18]** W. Shi_ et al._~~87~~,** [P19]**<br>K. Sullivan_ et al._94,** [P20]** R. Broderick20,** [P21]** N. Li_ et al._58,** [P22]** S. Russell_ et al._77,** [P23]** A.<br>Broggi_ et al._22,** [P24]** J. Schumann and S. Nelson84,** [P25]** J. Hull_ et al._41,** [P26]** L. Pulina and A.<br>Tacchella71,** [P27]** S. Lefevre_ et al._55|
|Iteration 2|**[P28]** X. Huang_ et al._~~40~~,** [P29]** K. Sullivan_ et al._~~94~~,** [P30]** P. Gupta P and J. Schumann~~32~~,** [P31]**<br>J. Schumann_ et al._81,** [P32]** R. Broderick19,** [P33]** Y. Liu_ et al._59,** [P34]** S. Yerramalla_ et al._106,<br>**[P35]** R. Zakrzewski109,** [P36]** S. Yerramalla_ et al._106,** [P37]** G. Katz_ et al._50,** [P38]** A. Akametalu<br>_et al._3,** [P39]** S. Seshia_ et al._85,** [P40]** A. Mili_ et al._63,** [P41]** Z. Kurd_ et al._53,** [P42]** L. Pulina and<br>A. Tacchella72,** [P43]** J. Schumann_ et al._84,** [P44]** R. Zakrzewski107,** [P45]** D. Mackall_ et al._60|
|Iteration 3|**[P46]** S. Jacklin_ et al._~~48~~,** [P47]** N. Nguyen and S. Jacklin~~67~~,** [P48]** J. Schumann and Y. Liu~~82~~,** [P49]**<br>N. Nguyen and S. Jacklin68,** [P50]** S. Jacklin_ et al._49,** [P51]** J. Taylor_ et al._98,** [P52]** G. Li_ et al._56,<br>**[P53]** P. Gupta_ et al._33,** [P54]** K. Scheibler_ et al._80,** [P55]** P. Gupta_ et al._31,** [P56]** S. Jacklin_ et_<br>_al._47,** [P57]** V. Cortellessa_ et al._25,** [P58]** S. Yerramalla_ et al._105|
|Iteration 4|**[P58]** S. Jacklin,~~46~~,** [P59]** F. Soares_ et al._~~92~~,** [P60]** X. Zhang_ et al._~~110~~,** [P61]** F. Soares and<br>J. Burken91,** [P62]** C. Torens_ et al._100,** [P63]** J. Bosworth and P. Williams-Hayes16,** [P64]** R.<br>Zakrzewski108|



M. C. Laghi, P. Medici, and P. Versari. Extensive
Tests of Autonomous Driving Technologies. _IEEE_
_Transactions on Intelligent Transportation Systems_,
14(3):1403–1415, 2013.
22. A. Broggi, P. Cerri, P. Medici, P. P. Porta, and G. Ghisio. Real Time Road Signs Recognition. In _2007 IEEE_
_Intelligent Vehicles Symposium_, pages 981–986, 2007.
23. A. Carvalho, S. Lefevre, G. Schildbach, J. Kong, and
F. Borelli. Automated driving: The role of forecasts
and uncertainty - A control perspective. _European_
_Journal of Control_, 2015.
24. Matthew Clark, Kris Kearns, Jim Overholt, Kerianne
Gross, Bart Barthelemy, and Cheryl Reed. Air Force
Research Laboratory Test and Evaluation, Verification and Validation of Autonomous Systems Challenge Exploration. Technical report, Air Force Research Lab Wright-Patterson, 2014.
25. Vittorio Cortellessa, Bojan Cukic, Diego Del Gobbo,
Ali Mili, Marcello Napolitano, Mark Shereshevsky,
and Harjinder Sandhu. Certifying Adaptive Flight
Control Software. In _Proc. of the Software Risk Man-_
_agement Conf._, 2000.
26. D. S. Cruzes and T. Dyba. Recommended Steps for
Thematic Synthesis in Software Engineering. In _In_
_Proc. of the International Symposium on Empirical_
_Software Engineering and Measurement_, pages 275–



284, 2011.
27. Jeff Donahue, Yangqing Jia, Oriol Vinyals, Judy Hoffman, Ning Zhang, Eric Tzeng, and Trevor Darrell. DeCAF: A Deep Convolutional Activation Feature for
Generic Visual Recognition. In _PMLR_, pages 647–
655, 2014.
28. F. Falcini, G. Lami, and A. Costanza. Deep Learning
in Automotive Software. _IEEE Software_, 34(3):56–
63, 2017.
29. Xavier Glorot, Antoine Bordes, and Yoshua Bengio.
Domain Adaptation for Large-scale Sentiment Classification: A Deep Learning Approach. In _Proc. of the_
_28th International Conference on International Con-_
_ference on Machine Learning_, pages 513–520, 2011.
30. Ian Goodfellow. NIPS 2016 Tutorial: Generative Adversarial Networks. _arXiv:1701.00160_, 2016.
31. P. Gupta, Ph D, K. A. Loparo, Ph D, D. Mackall,
J. Schumann, Ph D, and F. R. Soares. Verification and
Validation Methodology of Real-time Adaptive Neural Networks for Aerospace Applications. Technical
report, NASA, 2004.
32. P. Gupta and J. Schumann. A tool for verification
and validation of neural network based adaptive controllers for high assurance systems. In _Proc. of the_
_8th IEEE International Symposium on High Assur-_
_ance Systems Engineering_, pages 277–278, 2004.


_M. Borg et al. / Safely Entering the Deep_



33. Pramod Gupta, Kurt Guenther, John Hodgkinson,
Stephen Jacklin, Michael Richard, Johann Schumann,
and Fola Soares. Performance Monitoring and Assessment of Neuro-Adaptive Controllers for Aerospace
Applications Using a Bayesian Approach. In _AIAA_
_Guidance, Navigation, and Control Conference and_
_Exhibit_ . American Institute of Aeronautics and Astronautics, 2005.
34. A. Gurghian, T. Koduri, S. Bailur, K. Carey, and
V. Murali. DeepLanes: End-To-End Lane Position Estimation Using Deep Neural Networks. In _Proc. of_
_the IEEE Conference on Computer Vision and Pattern_
_Recognition Workshops_, pages 38–45, 2016.
35. Kaiming He, Xiangyu Zhang, Shaoqing Ren, and
Jian Sun. Delving Deep into Rectifiers: Surpassing
Human-Level Performance on ImageNet Classification. In _Proc. of the International Conference on Com-_
_puter Vision_, 2015.
36. Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian
Sun. Deep Residual Learning for Image Recognition.
In _Pro. of the IEEE Conference on Computer Vision_
_and Pattern Recognition_, pages 770–778, 2016.
37. Karl Heckemann, Manuel Gesell, Thomas Pfister,
Karsten Berns, Klaus Schneider, and Mario Trapp.
Safe Automotive Software. In _Knowledge-Based_
_and Intelligent Information and Engineering Systems_,
pages 167–176. Springer, Berlin, Heidelberg, 2011.
38. P. Helle, W. Schamai, and C. Strobel. Testing of Autonomous Systems - Challenges and Current State-ofthe-Art. In _Proc. of the 26th Annutal INCOSE Inter-_
_national Symposium_, pages 571–584, 2016.
39. Jens Henriksson, Markus Borg, and Cristofer Englund. Automotive safety and machine learning: Initial results from a study on how to adapt the iso 26262
safety standard. In _Proc. of the 1st International Work-_
_shop on Software Engineering for AI in Autonomous_
_Systems_, pages 47–49. IEEE, 2018.
40. Xiaowei Huang, Marta Kwiatkowska, Sen Wang, and
Min Wu. Safety Verification of Deep Neural Networks. _arXiv:1610.06940_, 2016.
41. J. Hull, D. Ward, and R. R. Zakrzewski. Verification
and validation of neural networks for safety-critical
applications. In _Proc. of the 2002 American Control_
_Conference_, volume 6, pages 4789–4794, 2002.
42. _{_ IEEE Computer Society _}_ . 610.12-1990 IEEE standard glossary of software engineering terminology.
Technical report, 1990.
43. _{_ International Electrotechnical Commission _}_ . _IEC_
_61508 ed 1.0, Electrical/electronic/programmable_
_electronic safety-related systems_ . 2010.
44. _{_ International Organization for Standardization _}_ . _ISO_
_26262 Road vehicles - Functional safety_ . 2011.
45. Martin Ivarsson and Tony Gorschek. A method for
evaluating rigor and industrial relevance of technology evaluations. _Empirical Software Engineering_,



16(3):365–395, 2011.
46. Stephen Jacklin. Closing the Certification Gaps in
Adaptive Flight Control Software. In _AIAA Guid-_
_ance, Navigation and Control Conference and Ex-_
_hibit_ . American Institute of Aeronautics and Astronautics, 2008.
47. Stephen Jacklin, Johann Schumann, Pramod Gupta,
M Lowry, John Bosworth, Eddie Zavala, Kelly Hayhurst, Celeste Belcastro, and Christine Belcastro. Verification, Validation, and Certification Challenges for
Adaptive Flight-Critical Control System Software. In
_AIAA Guidance, Navigation, and Control Conference_
_and Exhibit_ . 2004.
48. Stephen Jacklin, Johann Schumann, Pramod Gupta,
Michael Richard, Kurt Guenther, and Fola Soares.
Development of Advanced Verification and Validation Procedures and Tools for the Certification of
Learning Systems in Aerospace Applications. In _In-_
_fotech@Aerospace_ . American Institute of Aeronautics
and Astronautics, 2005.
49. Stephen A. Jacklin, Johann Schumann, John T.
Bosworth, Peggy S. Williams-Hayes, and Richard S.
Larson. Case Study: Test Results of a Tool and
Method for In-Flight, Adaptive Control System Verification on a NASA F-15 Flight Research Aircraft.
In _Proc. of the 7th World Congress on Computa-_
_tional Mechanics Minisymposium: Accomplishments_
_and Challenges in Verification and Validation_, 2006.
50. Guy Katz, Clark Barrett, David Dill, Kyle Julian,
and Mykel Kochenderfer. Reluplex: An Efficient
SMT Solver for Verifying Deep Neural Networks.
_arXiv:1702.01135_, 2017.
51. Alessia Knauss, Jan Schroeder, Christian Berger, and
Henrik Eriksson. Software-related Challenges of Testing Automated Vehicles. In _Proc. of the 39th Interna-_
_tional Conference on Software Engineering Compan-_
_ion_, pages 328–330, 2017.
52. Matthias Korte, Frdric Holzmann, Gerd Kaiser, Volker
Scheuch, and Hubert Roth. Design of a Robust Plausibility Check for an Adaptive Vehicle Observer in an
Electric Vehicle. In _Advanced Microsystems for Auto-_
_motive Applications 2012_, pages 109–119. Springer,
Berlin, Heidelberg, 2012.
53. Zeshan Kurd, Tim Kelly, and Jim Austin. Developing artificial neural networks for safety critical systems. _Neural Computing and Applications_, 16(1):11–
19, 2007.
54. Yann LeCun, Yoshua Bengio, and Geoffrey Hinton.
Deep learning. _Nature_, 521(7553):436–444, 2015.
55. Stephanie Lefevre, Dizan Vasquez, and Christian
Laugier. A survey on motion prediction and risk assessment for intelligent vehicles. _ROBOMECH Jour-_
_nal_, 1(1):1–14, 2014.
56. G. Li, M. Lu, and B. Liu. A Scenario-Based Method
for Safety Certification of Artificial Intelligent Soft

_M. Borg et al. / Safely Entering the Deep_



ware. In _Proc. of the 2010 International Conference_
_on Artificial Intelligence and Computational Intelli-_
_gence_, volume 3, pages 481–483, 2010.
57. L. Li, W. L. Huang, Y. Liu, N. N. Zheng, and F. Y.
Wang. Intelligence Testing for Autonomous Vehicles:
A New Approach. _IEEE Transactions on Intelligent_
_Vehicles_, 1(2):158–166, 2016.
58. Nan Li, Dave Oyler, Mengxuan Zhang, Yildiray
Yildiz, Ilya Kolmanovsky, and Anouck Girard. GameTheoretic Modeling of Driver and Vehicle Interactions
for Verification and Validation of Autonomous Vehicle Control Systems. _arXiv:1608.08589_, 2016.
59. Yan Liu, Bojan Cukic, and Srikanth Gururajan. Validating neural network-based online adaptive systems:
a case study. _Software Quality Journal_, 15(3):309–
326, 2007.
60. D. Mackall, S. Nelson, and J. Schumman. Verification and validation of neural networks for aerospace
systems. Technical report, 2002.
61. D. Maji, A. Santara, P. Mitra, and D. Sheet. Ensemble of Deep Convolutional Neural Networks for
Learning to Detect Retinal Vessels in Fundus Images.
_arXiv:1603.04833_, 2016.
62. E. Mhamdi, Rachid Guerraoui, and Sebastien
Rouault. On The Robustness of a Neural Network.
_arXiv:1707.08167_, 2017.
63. A. Mili, GuangJie Jiang, B. Cukic, Yan Liu, and R. B.
Ayed. Towards the verification and validation of online learning systems: general framework and applications. In _Proc. of the 37th Annual Hawaii Interna-_
_tional Conference on System Sciences_, 2004.
64. Niklas Moller and Sven Ove Hansson. Principles of
engineering safety: Risk and uncertainty reduction.
_Reliability Engineering & System Safety_, 93(6):798–
805, 2008.
65. Ahmad Mozaffari, Mahyar Vajedi, and Nasser L.
Azad. A robust safety-oriented autonomous cruise
control scheme for electric vehicles based on model
predictive control and online sequential extreme learning machine with a hyper-level fault tolerance-based
supervisor. _Neurocomputing_, 151:845 – 856, 2015.
66. _{_ National Instruments _}_ . What is the ISO 26262 Functional Safety Standard?
67. Nhan T. Nguyen and Stephen A. Jacklin. Neural Net
Adaptive Flight Control Stability. In _Verification and_
_Validation Challenges, and Future Research, IJCNN_
_Conference_, 2007.
68. Nhan T. Nguyen and Stephen A. Jacklin. Stability,
Convergence, and Verification and Validation Challenges of Neural Net Adaptive Flight Control. In Johann Schumann and Yan Liu, editors, _Applications of_
_Neural Networks in High Assurance Systems_, pages
77–110. Springer Berlin Heidelberg, Berlin, Heidelberg, 2010.
69. Maxime Oquab, Leon Bottou, Ivan Laptev, and Josef



Sivic. Learning and Transferring Mid-Level Image Representations using Convolutional Neural Networks. In _Proc. of the IEEE Conference on Com-_
_puter Vision and Pattern Recognition_, pages 1717–
1724, 2014.
70. Marco AF Pimentel, David A Clifton, Lei Clifton, and
Lionel Tarassenko. A review of novelty detection. _Sig-_
_nal Processing_, 99:215–249, 2014.
71. Luca Pulina and Armando Tacchella. An AbstractionRefinement Approach to Verification of Artificial
Neural Networks. In _Computer Aided Verification_,
pages 243–257. Springer, Berlin, Heidelberg, 2010.
72. Luca Pulina and Armando Tacchella. NeVer: a tool
for artificial neural networks verification. _Annals of_
_Mathematics and Artificial Intelligence_, 62(3):403–
425, 2011.
73. Laura Pullum, Brian Taylor, and Majorie Darrah.
_Guidance for the Verification and Validation of Neural_
_Networks_ . John Wiley & Sons, Inc., 2007.
74. Alec Radford, Luke Metz, and Soumith Chintala. Unsupervised Representation Learning with
Deep Convolutional Generative Adversarial Networks. _arXiv:1511.06434_, 2015.
75. Sebastian Ramos, Stefan Gehrig, Peter Pinggera, Uwe
Franke, and Carsten Rother. Detecting Unexpected
Obstacles for Self-Driving Cars: Fusing Deep Learning and Geometric Modeling. _arXiv:1612.06573_,
2016.
76. Louis M. Rea and Richard A. Parker. _Designing_
_and Conducting Survey Research: A Comprehensive_
_Guide_ . John Wiley & Sons, 4 edition, 2014.
77. S. Russell, D. Dewey, and M. Tegmark. Research
Priorities for Robust and Beneficial Artificial Intelligence. _arXiv:1602.03506_, 2016.
78. Rick Salay, Rodrigo Queiroz, and Krzysztof Czarnecki. An Analysis of ISO 26262: Using Machine Learning Safely in Automotive Software.
_arXiv:1709.02435_, 2017. arXiv: 1709.02435.
79. Ahmad Sallab, Mohammed Abdou, Etienne Perot,
and Senthil Yogamani. End-to-End Deep Reinforcement Learning for Lane Keeping Assist.
_arXiv:1612.04340_, 2016.
80. Karsten Scheibler, Leonore Winterer, Ralf Wimmer,
and Bernd Becker. Towards Verification of Artificial
Neural Networks. In _Proc. of MBMV_, 2015.
81. J. Schumann, P. Gupta, and S. Jacklin. Toward Verification and Validation of Adaptive Aircraft Controllers. In _Proc. of the IEEE Aerospace Conference_,
pages 1–6, 2005.
82. J. Schumann and Y. Liu. Tools and Methods for the
Verification and Validation of Adaptive Aircraft Control Systems. In _Proc. of the IEEE Aerospace Confer-_
_ence_, pages 1–8, 2007.
83. Johann Schumann, Pramod Gupta, and Yan Liu. Application of Neural Networks in High Assurance Sys

_M. Borg et al. / Safely Entering the Deep_



tems: A Survey. In _Applications of Neural Networks_
_in High Assurance Systems_, Studies in Computational
Intelligence, pages 1–19. Springer, Berlin, Heidelberg, 2010.
84. Johann Schumann and Stacy Nelson. Toward V&V
of Neural Network Based Controllers. In _Proc. of the_
_1st Workshop on Self-healing Systems_, pages 67–72,
2002.
85. Sanjit A. Seshia, Dorsa Sadigh, and S. Shankar Sastry. Formal Methods for Semi-autonomous Driving.
In _Proc. of the 52nd Annual Design Automation Con-_
_ference_, pages 148:1–148:5, 2015.
86. Sanjit A. Seshia, Dorsa Sadigh, and S. Shankar
Sastry. Towards Verified Artificial Intelligence.
_arXiv:1606.08514_, 2016.
87. Weijing Shi, Mohamed Baker Alawieh, Xin Li,
Huafeng Yu, Nikos Arechiga, and Nobuyuki Tomatsu.
Efficient Statistical Validation of Machine Learning
Systems for Autonomous Driving. In _Proc. of the 35th_
_International Conference on Computer-Aided Design_,
pages 36:1–36:8, 2016.
88. Karen Simonyan and Andrew Zisserman. Very
Deep Convolutional Networks for Large-Scale Image
Recognition. _arXiv:1409.1556_, 2014.
89. Sayanan Sivaraman and Mohan M. Trivedi. Active
Learning for On-road Vehicle Detection: A Comparative Study. _Mach. Vision Appl._, 25(3):599–611, 2014.
90. Leon Sixt, Benjamin Wild, and Tim Landgraf.
RenderGAN: Generating Realistic Labeled Data.
_arXiv:1611.01331_, 2016.
91. F. Soares and J. Burken. A Flight Test Demonstration of On-line Neural Network Applications in Advanced Aircraft Flight Control System. In _Proc. of_
_the 2006 International Conference on Computational_
_Inteligence for Modelling Control and Automation_
_and International Conference on Intelligent Agents_
_Web Technologies and International Commerce_, pages
136–136, 2006.
92. Fola Soares, John Burken, and Tshilidzi Marwala.
Neural Network Applications in Advanced Aircraft
Flight Control System, a Hybrid System, a Flight Test
Demonstration. In Irwin King, Jun Wang, Lai-Wan
Chan, and DeLiang Wang, editors, _Neural Informa-_
_tion Processing_, pages 684–691, Berlin, Heidelberg,
2006. Springer Berlin Heidelberg.
93. B. Spanfelner, D. Richter, S. Ebel, U. Wilhelm,
W. Branz, and C. Patz. Challenges in applying the
ISO 26262 for driver assistance. In _Proc. of the Schw-_
_erpunkt Vernetzung, 5. Tagung Fahrerassistenz_, Munich, Germany, 2012.
94. K. B. Sullivan, K. M. Feigh, F. T. Durso, U. Fischer,
V. L. Pop, K. Mosier, J. Blosch, and D. Morrow. Using neural networks to assess human-automation interaction. In _2011 IEEE/AIAA 30th Digital Avionics_
_Systems Conference_, pages 6A4–1–6A4–10, 2011.



95. Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich.
Going Deeper With Convolutions. In _Proc. of the_
_IEEE Conference on Computer Vision and Pattern_
_Recognition_, pages 1–9, 2015.
96. Brian J. Taylor. _Methods and Procedures for the Ver-_
_ification and Validation of Artificial Neural Networks_ .
Springer Science & Business Media, 2006. GoogleBooks-ID: ax3Q ~~Y~~ BuXFEC.
97. Brian J. Taylor, Marjorie A. Darrah, and Christina D.
Moats. Verification and validation of neural networks: a sampling of research in progress. In _Proc._
_of SPIE 5103, Intelligent Computing: Theory and Ap-_
_plications_, volume 5103, pages 8–16, 2003.
98. J. Taylor, E. Yudkowsky, P. LaVictoire, and A. Critch.
Alignment for advanced machine learning systems.
Technical report, Machine Intelligence Research Institute, 2016.
99. Yuchi Tian, Kexin Pei, Suman Jana, and Baishakhi
Ray. Deeptest: Automated testing of deep-neuralnetwork-driven autonomous cars. In _Proc. of the 40th_
_International Conference on Software Engineering_,
pages 303–314. ACM, 2018.
100. Christoph Torens, Florian-M. Adolf, and Lukas Goormann. Certification and Software Verification Considerations for Autonomous Unmanned Aircraft. _Journal_
_of Aerospace Information Sys._, 11(10):649–664, 2014.
101. K. Varshney. Engineering safety in machine learning.
In _Proc. of the 2016 Information Theory and Appl._
_Workshop_, pages 1–5, January 2016.
102. K. Varshney, R. Prenger, T. Marlatt, B. Chen, and
W. Hanley. Practical Ensemble Classification Error Bounds for Different Operating Points. _IEEE_
_Transactions on Knowledge and Data Engineering_,
25(11):2590–2601, 2013.
103. C. Wohlin. Guidelines for Snowballing in Systematic
Literature Studies and a Replication in Software Engineering. In _Proc. of the 18th International Conference_
_on Evaluation and Assessment in Software Engineer-_
_ing_, pages 38:1–38:10, 2014.
104. X. Yao and Y. Liu. A new evolutionary system for
evolving artificial neural networks. _IEEE Transac-_
_tions on Neural Networks_, 8(3):694–713, 1997.
105. Sampath Yerramalla, Edgar Fuller, Martin Mladenovski, and Bojan Cukic. Lyapunov Analysis of Neural
Network Stability in an Adaptive Flight Control System. In _Proc. of the 6th International Conference on_
_Self-stabilizing Systems_, pages 77–92, 2003.
106. Sampath Yerramalla, Yan Liu, Edgar Fuller, Bojan
Cukic, and Srikanth Gururajan. An Approach to
V&V of Embedded Adaptive Systems. In Michael G.
Hinchey, James L. Rash, Walter F. Truszkowski, and
Christopher A. Rouff, editors, _Formal Approaches to_
_Agent-Based Systems_, pages 173–188, Berlin, Heidel

_M. Borg et al. / Safely Entering the Deep_



berg, 2005. Springer Berlin Heidelberg.
107. R. R. Zakrzewski. Verification of a trained neural network accuracy. In _Proc. of the International Joint_
_Conference on Neural Networks_, volume 3, pages
1657–1662, 2001.
108. R. R. Zakrzewski. Verification of performance of a
neural network estimator. In _Proc. of the 2002 Inter-_
_national Joint Conference on Neural Networks_, volume 3, pages 2632–2637, 2002.
109. R. R. Zakrzewski. Randomized approach to verification of neural networks. In _Proc. of the IEEE Inter-_
_national Joint Conference on Neural Networks_, volume 4, pages 2819–2824, 2004.
110. Xiaodong Zhang, Matthew Clark, Kudip Rattan, and
Jonathan Muse. Controller Verification in Adaptive
Learning Systems Towards Trusted Autonomy. In
_Proc. of the ACM/IEEE 6th International Conference_



_on Cyber-Physical Systems_, pages 31–40, 2015.
111. H. Zhu, K. Yuen, L. Mihaylova, and H. Leung.
Overview of Environment Perception for Intelligent
Vehicles. _IEEE Transactions on Intelligent Trans-_
_portation Systems_, 18(10):2584–2601, 2017.
112. X. Zou, R. Alexander, and J. McDermid. On the Validation of a UAV Collision Avoidance System Developed by Model-Based Optimization: Challenges and
a Tentative Partial Solution. In _Proc. of the 46th An-_
_nual IEEE/IFIP International Conference on Depend-_
_able Systems and Networks Workshop_, pages 192–
199, 2016.
113. Xueyi Zou, Rob Alexander, and John McDermid.
Safety Validation of Sense and Avoid Algorithms Using Simulation and Evolutionary Search. In _Com-_
_puter Safety, Reliability, and Security_, pages 33–48.
Springer, Cham, 2014.


