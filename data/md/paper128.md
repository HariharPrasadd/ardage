**Representation Bias in Data: A Survey on Identification and Resolution**

**Techniques**


NIMA SHAHBAZI, University of Illinois Chicago, USA

YIN LIN, University of Michigan, USA

ABOLFAZL ASUDEH, University of Illinois Chicago, USA

H. V. JAGADISH, University of Michigan, USA


Data-driven algorithms are only as good as the data they work with, while data sets, especially social data, often fail to represent


minorities adequately. Representation Bias in data can happen due to various reasons ranging from historical discrimination to


selection and sampling biases in the data acquisition and preparation methods. Given that “bias in, bias out”, one cannot expect


AI-based solutions to have equitable outcomes for societal applications, without addressing issues such as representation bias. While


there has been extensive study of fairness in machine learning models, including several review papers, bias in the data has been less


studied. This paper reviews the literature on identifying and resolving representation bias as a feature of a data set, independent of


how consumed later. The scope of this survey is bounded to structured (tabular) and unstructured (e.g., image, text, graph) data. It


presents taxonomies to categorize the studied techniques based on multiple design dimensions and provides a side-by-side comparison


of their properties.


There is still a long way to fully address representation bias issues in data. The authors hope that this survey motivates researchers


to approach these challenges in the future by observing existing work within their respective domains.


CCS Concepts: • **Information systems** → **Data management systems** ;


Additional Key Words and Phrases: Responsible Data Science, Fairness in Machine Learning, Data Equity Systems, Data-centric AI,


AI-Ready Data


**ACM Reference Format:**


Nima Shahbazi, Yin Lin, Abolfazl Asudeh, and H. V. Jagadish. 2021. Representation Bias in Data: A Survey on Identification and


Resolution Techniques. In _Woodstock ’18: ACM Symposium on Neural Gaze Detection, June 03–05, 2018, Woodstock, NY._ ACM, New


[York, NY, USA, 47 pages. https://doi.org/10.1145/1122445.1122456](https://doi.org/10.1145/1122445.1122456)


**1** **INTRODUCTION**


Data-driven decision-making shapes every corner of human life, from autonomous vehicles to healthcare and even


predictive policing and criminal sentencing. A critical question, particularly in applications impacting human beings,


is how trustworthy the decision made by the system is. It is easy to see that the accuracy of a data-driven decision


depends, first and foremost, on the data used to make it. After all, the system learns the phenomena that data represent.


As a first step, we may desire that the data should represent the underlying data distribution from which the production


data will be drawn. But that is not enough since it only tells us about the overall model performance. Although a system


may generally perform well in terms of accuracy, it could fail for less populated regions in the data with insufficient


Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not

made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components

of this work owned by others than ACM must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to

redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.


© 2021 Association for Computing Machinery.

Manuscript submitted to ACM


1


Woodstock ’18, June 03–05, 2018, Woodstock, NY Nima Shahbazi, Yin Lin, Abolfazl Asudeh, and H. V. Jagadish


representation. These regions may matter because they frequently represent some minority (sub)population in society.


They could also represent cases that may not happen very often but have a relevant impact on the correctness of a


critical decision. In short, if data is not representative of a given population, the outcome of the decision system for that


subpopulation may not be trustworthy.


_Representation Bias_ happens when the training data under-represents (and subsequently fails to generalize well) some


parts of the target population [ 156 ]. Data representation bias can originate from how (and from where) the data was


originally collected or be caused by the biases introduced after collection, either historically, cognitively, or statistically.


Representation bias can happen due to selection bias, i.e. when the sampling method only reaches a portion of the


population or the population of interest has changed or is distinct from the population used during model training.


For example, a survey to measure the illegal drug use of teenagers could be biased if it only includes high school


students and ignores home-schooled students or dropouts. Another potential reason is the skewness of the underlying


distribution. Suppose the target population for a particular medical data set is adults aged 18-60. There are minority


groups within this population: for example, pregnant people may make up only 5% of the target population. Even with


perfect sampling and an identical population, the model is prone to be less robust for the group of pregnant people


because it has fewer data points to learn from [ 156 ]. Furthermore, even if we carefully arrange for uniform sampling by


age, we may find that sampling is non-uniform for pregnant people. For example, there may be proportionately fewer


pregnant people over 40. If some group is a minority in the underlying distribution, then even random sampling will


not help the under-representation issue for this group.


Representation bias is almost always guaranteed without a systematic approach to data collection. For example, in a


survey data collection, a crucial step is to identify all the sub-populations in the underlying distribution based on the


desired demographic information and ensure that the survey reaches all of them while enough samples are collected


from each. However, the problem is that data scientists usually do not have any control over the data collection process,


resulting in the utilization of “found data” in most data-driven decision-making systems. Therefore, with no guarantee


on the aforementioned steps in the data collection process, the found data is most likely a biased sample.


Representation bias in data is not a new problem and has been a known issue in data mining, database management,


and statistics communities. There is a rich line of work on the problem of discovering interesting patterns, regularities,


or finding empty space in the data that is a parallel and relatively similar problem to identifying representation bias


in data sets [ 57, 111, 115, 116 ]. However, with the emergence of responsible data science and trustworthy AI, this


problem has been addressed with greater vigor and from a brand new perspective in recent years. This survey discusses


techniques for identifying and resolving representation bias in data sets, introducing taxonomies to classify these


techniques based on multiple dimensions. Note that while the literature on algorithmic fairness is primarily concerned


with promoting fairness in machine learning (ML) _models_, bias is sought to be addressed in the _data sets_, regardless of


how the data is ultimately consumed.


We start the paper by presenting a big-picture overview of the _fairness literature_ in Section 2. This will help us specify


the scope of this survey w.r.t. fairness approaches and existing surveys. Next, in Section 3, we zoom in on the notion of


_representation bias_, explaining the reasons that give rise to it, and presenting techniques for measuring representation


bias. In Section 4, we propose a taxonomy to categorize different approaches to identify and resolve representation


bias in _structured data_ based on factors such as objectives and capabilities. Following our taxonomy’s guidelines, we


investigate each work’s details, explain its novelty, and discuss its pros and cons. In Section 5, we review the techniques


for identifying and resolving representation bias in _unstructured data_ such as images, text, speech, and graphs. Finally,


2


Representation Bias in Data: A Survey on Identification and Resolution Techniques Woodstock ’18, June 03–05, 2018, Woodstock, NY


in Section 6, we present an overview of the reviewed works and conclude the survey by discussing aspects that have


been less noticed in the existing lines of work and propose some possible directions for the researchers to investigate.


**2** **AN OVERVIEW OF FAIRNESS LITERATURE**


As AI replaces human beings in various critical fields, the topic of fairness among the affected population becomes


more crucial. In recent years, the general topic of fairness has drawn sizable attention from different communities,


specifically in the ML field. Many surveys [ 18, 34, 121, 137, 152 ] and tutorials have been published on the related topics

and even the conference ACM FAccT [1] has been dedicated to this topic. Before focusing on representation bias in a data


set, it is beneficial to review the big picture of fairness literature, including the definitions and techniques to achieve


fairness. Given this context, we will specify the scope of this survey.


**2.1** **Definitions of Fairness**


There is no clear agreement on the definitions of fairness since it all depends on the task we target to solve and the


numerous kinds of bias that can exist in data. However, at a high level, fairness definitions can be viewed from three


perspectives [19]: _individual fairness_, _group fairness_, _subgroup fairness_ .


_**Individual Fairness**_ _._ Individual fairness is the most granular notion of fairness, requiring similar outcomes for


similar individuals [56].


_**Group Fairness**_ _._ Group fairness is the most popular category of fairness definitions for learning models. The term


“group” refers to the classification of individuals within a population into a particular social category that has been


historically subject to discriminatory treatment [ 19 ]. Examples of such social categories a.k.a. **sensitive attributes**


include _race, gender, sexual orientation, age, religion, disability_, etc. A model satisfies some group fairness definition if it


has equal or similar performance on different groups w.r.t. the associated fairness measures. Most of ML group fairness


metrics could be classified into the following categories [11, 19]: _independence_, _separation_, _sufficiency_, _causation_ .


_Independence_ only relies on the model’s predicted outcome, and a model satisfies independence if its outcome is


independent of the sensitive attributes. Let _ℎ_ ( _𝑥_ ) and G represent the model outcome and the demographic groups,

respectively. Under Independence measures [2],


_ℎ_ ( _𝑥_ ) ⊥⊥G (1)


Measures such as _Statistical Parity_ [ 56 ] fall under this category. These measures indicate that different demographic


groups have (almost) equal probabilities to generate positive (favorable) prediction: ∀ _𝑔_ _𝑖_ _,𝑔_ _𝑗_ ∈G _, 𝑃𝑟_ ( _ℎ_ ( _𝑥_ ) = 1 | _𝑔_ _𝑖_ ) ≃

_𝑃𝑟_ ( _ℎ_ ( _𝑥_ ) = 1 | _𝑔_ _𝑗_ ) . _Conditional statistical parity_ [ 46 ] extends the definition of independence by considering a set of

legitimate attributes _𝐿_ that could affect the outcome: ∀ _𝑔_ _𝑖_ _,𝑔_ _𝑗_ ∈G _, 𝑃𝑟_ ( _ℎ_ ( _𝑥_ ) = 1 | _𝑥_ _𝑙_ = _𝑙,𝑔_ _𝑖_ ) ≃ _𝑃𝑟_ ( _ℎ_ ( _𝑥_ ) = 1 | _𝑥_ _𝑙_ = _𝑙,𝑔_ _𝑗_ ) . For


example, suppose the demographic groups are male and female, and the legitimate factor is marital status. Therefore,


the probability of married male and married female getting a positive prediction result should be equivalent.


_Separation_ is satisfied when the outcome of the model is independent of the sensitive attribute(s) conditioned on the


ground-truth label _𝑦_ . That is,
� _ℎ_ ( _𝑥_ ) ⊥⊥G��� _𝑦_ (2)


1 [https://facctconference.org/](https://facctconference.org/)
2 ⊥⊥ is the mathematical independence operation between two random variables.


3


Woodstock ’18, June 03–05, 2018, Woodstock, NY Nima Shahbazi, Yin Lin, Abolfazl Asudeh, and H. V. Jagadish


Two well-known measures in this category are _Equalized Odds_ and _Equal Opportunity_ [ 79 ]. Equalized odds is considered


in contexts that correctly predicting positive outcomes and minimizing costly false positives are both of high importance:


∀ _𝑔_ _𝑖_ ∈G _, 𝑃𝑟_ ( _ℎ_ ( _𝑥_ ) = 1 | _𝑔_ _𝑖_ _,𝑦_ = 1 ) ≃ _𝑃𝑟_ ( _ℎ_ ( _𝑥_ ) = 1 | _𝑦_ = 1 ) and _𝑃𝑟_ ( _ℎ_ ( _𝑥_ ) = 1 | _𝑔_ _𝑖_ _,𝑦_ = 0 ) ≃ _𝑃𝑟_ ( _ℎ_ ( _𝑥_ ) = 1 | _𝑦_ = 0 ) . Equal


opportunity is a reasonable measure when predicting the positive outcome correctly is crucial and false positives are


not costly: ∀ _𝑔_ _𝑖_ ∈G _, 𝑃𝑟_ ( _ℎ_ ( _𝑥_ ) = 1| _𝑔_ _𝑖_ _,𝑦_ = 1) ≃ _𝑃𝑟_ ( _ℎ_ ( _𝑥_ ) = 1| _𝑦_ = 1).


_Sufficiency_, on the other hand, is satisfied if, under the same model outcomes, sensitive attribute(s) and the true


outcome are independent. That is,
� _ℎ_ ( _𝑥_ ) ⊥⊥ _𝑦_ ��� G (3)


Sufficiency can be measured with _Predictive Parity_ [ 43 ]. Positive predictive parity guarantees an equal chance of success,


given the positive prediction for all subgroups: ∀ _𝑔_ _𝑖_ ∈G _, 𝑃𝑟_ ( _𝑦_ = 1 | _ℎ_ ( _𝑥_ ) = 1 _,𝑔_ _𝑖_ ) ≃ _𝑃𝑟_ ( _𝑦_ = 1 | _ℎ_ ( _𝑥_ ) = 1 ) . Similarly, negative


predictive parity ensures an equal chance of success given the negative prediction for all subgroups.


_Causation_, aka counterfactual fairness [ 105, 143 ] focuses on the causal relationship between attributes, for instance


when an attribute _𝐴_ affects attribute _𝐵_, which in turn affects attribute _𝐶_ . The counterfactual definition of fairness


follows the intuition that a decision is fair for an individual if, in a counterfactual world, the decision would not change


had the individual belonged to a different demographic group.


Please note that this is not an exhaustive list of group fairness definitions, and we only introduced the ones more


commonly known and practiced. For a more exhaustive list and extensive discussion, please see [19, 162].


_**Subgroup Fairness**_ _._ Falling in between individual and group fairness, subgroup fairness [ 99 ] (also known as


intersectional fairness) metrics measure fairness (according to the above definitions) when groups are defined over the


intersection of values of multiple sensitive attributes (e.g. white male, white female, black male, and black female ).


Having discussed the existence of unfairness in ML models with the assistance of fairness definitions, next, we


introduce strategies to promote fairness.


**2.2** **Interventions to Achieve Fairness**


Fairness can be considered by ML models [ 35, 47 ] at different stages of the data analysis pipeline, shown in Figure 1. As


highlighted in the figure, the intervention strategies to achieve model fairness fall under three categories: _Pre-process_,


_In-process_, and _Post-process_ interventions.


_**Pre-process interventions**_ _._ The main idea of this category of techniques is to modify the data before feeding it into


the ML algorithms. The common pre-process interventions include: _data massaging_, _reweighting_, _sampling_, _modifying_


_feature representations_, _adversarial learning_, and _causal methods_ .


_Data massaging_, first proposed by Kamiran et al. [ 91 ], aims to select the best candidates in the training data for


relabeling by ranking the candidates according to their probability of belonging to the opposite class using a Naive


Bayesian classifier.


_Data reweighting_ [ 31 ] carefully assigns the tuples in the training set with different weights such that the new


distribution is discrimination free with respect to the sensitive attributes.


_Sampling_ methods [ 92 ] can be used to under or over-sample the training data set for the ML algorithms that


cannot directly work with weight. Given a sensitive attribute and considering the attribute value and label selection,


there are four groups: two need over-sampling, and the other two need under-sampling. The employed sampling


techniques include _uniform sampling_, which applies uniform probability to increase or decrease the size of the groups,


and _preferential sampling_, where borderline objects get higher priority to be duplicated or ignored.


4


Representation Bias in Data: A Survey on Identification and Resolution Techniques Woodstock ’18, June 03–05, 2018, Woodstock, NY


Fig. 1. Illustration of bias and fairness in data analytics pipeline (the pipeline is adapted from [85])


_Modifying feature representations_ includes learning an intermediate representation that maintains all the essential


information while removing any sign of the sensitive attribute. In [ 56 ], to handle individual fairness, Dwork et al. propose


to find a mapping from individuals to an intermediate representation that minimizes the loss subject while satisfying


the Lipschitz condition to guarantee that similar individuals are treated similarly. Further, to produce out-of-sample


representations to handle unseen examples, Zemel et al. [ 170 ] develop a learning approach to achieve both group and


individual fairness for ML models. The primary purpose is to learn a set of intermediate representations that satisfy


two goals: first, they encode the data as well as possible, and second, they should be blind to whether the individual is


from the protected group. The authors design a learning objective considering statistical parity, prediction accuracy,


and data loss. iFair [ 108 ] is a learning-to-rank algorithm that introduces a method for probabilistically mapping user


records into a low-rank representation to achieve individual fairness. Compared to the other learning representation


algorithms, it is agnostic to downstream machine learning algorithms and can handle a broader range of applications.


In [ 107 ], Lahoti et al. propose a method to model information of equally deserving individuals as a fairness graph.


Based on the fairness graph, the proposed method learns a Fair Representation (PFR) to capture both data-driven


similarities between individuals and pairwise side-information. Compared to the previous works for resolving individual


fairness, PFR avoids the most challenging part of eliciting a quantitative measure of similarity from human experts.


Optimized pre-processing [ 54 ] formulates an optimization problem to probabilistically transform the data to trade off


discrimination control, data utility, and individual fairness.


_Adversarial learning_ is another approach to increase the amount of data for the sensitive groups to achieve group


fairness [ 9, 167 ]. There are also generative models to enhance the training data set of fair classification. FairGAN [ 167 ]


uses a generative adversarial network (GAN) to generate synthetic data to enhance group fairness when the original


data is limited. It considers data utility, data fairness, classification utility, and classification fairness as important


requirements for the generated data.


_Causal methods_ uncover the causal relationships in the data and focus on the dependencies between sensitive


attributes and attributes acting as a proxy [ 44, 67, 72, 143 ]. In this regard, training data repairing strategies have been


suggested, such as [ 143 ] by Salimi et al., to minimally modify the databases by _remove, insert, update_ operations based


on the notion of conditional independence between outcome and the sensitive attributes.


_**In-process interventions**_ _._ In-processing methods mainly reinforce fairness by inducing constraints or adding


regularization terms to the objective function of the learning algorithm [ 10, 21, 32, 74, 94, 165, 169 ]. The enforced


constraints ensure that the algorithm treats different subpopulations equally w.r.t. the specified fairness measures. Other


5


Woodstock ’18, June 03–05, 2018, Woodstock, NY Nima Shahbazi, Yin Lin, Abolfazl Asudeh, and H. V. Jagadish


in-processing approaches include adversarial learning [ 22, 23, 36, 58, 163, 166, 171 ], re-weighing [ 87, 104, 172 ], and


bandits [ 60, 71, 89, 90, 118 ] approaches. Adversarial approaches use fairness measures to provide feedback to the model


by penalizing it if the sensitive attributes are predictable from any of the remaining attributes. This is usually achieved


by subjecting the model to many constraints and formulating the problem as a multiple-constraint optimization problem.


In-process re-weighting approaches usually begin by learning an unweighted classifier on the data and then using the


learned weights of the samples to retrain the classifier. Bandit-based approaches usually cannot define what it means


to be fair, but they may be able to recognize it when it is observed. This is usually achieved through the notions of


individual fairness.


_**Post-process interventions**_ _._ Post-processing methods manipulate the results of a classifier to promote fairness


among different groups. Hardt et al. [ 80 ] propose a post-processing technique to guarantee equalized odds by formulating


it as an optimization problem that finds the probabilities that can be used to change the output labels to remove


discrimination from protected groups. Calibrated equalized odds by Pleiss et al. [ 138 ] explores the relationship between


calibration and error rates. They provide an algorithm that aims to effectively find the unique feasible solution to


satisfy both by determining probabilities used to flip the output labels. Reject option classification by Kamiran et al.


[ 93 ] invokes a reject option and labels instances in deprived and favored groups by the posterior probability to reduce


discrimination.


**2.3** **Scope of the Survey**


Having discussed the intervention approaches to achieve model fairness, let us look at the pipeline of data analytics in


Figure 1 again. The model is often considered “the product” of the pipeline. Indeed, works on fairness focus on the


model. However, the _data set_ is also a product, of possible interest in its own right, in addition to its influence on model


fairness. Given a data set to train a model, fairness intervention techniques aim to build a model based on some fairness


criteria. On the other hand, this survey focuses on the other product of the pipeline, i.e., data sets, studying _bias as a_


_feature of a data set_, independent of how it is later consumed. In particular, the scope of the studies reviewed in this


survey is bounded to _representation bias in structured (tabular) and unstructured (image, graph, text, speech) data_ .


**2.4** **Related Surveys and Tutorials**


To the best of our knowledge, this is the first survey that specifically focuses on identifying and resolving representation


bias in a variety of structured and unstructured types of data from a data-centric standpoint. However, we would like to


highlight the existing surveys and tutorials on the general topics of bias and fairness and task-specific, data-specific, or


bias-specific approaches to debiasing and promoting fairness while pointing out how their scope differentiates from our


work.


Balayn et al. [ 18 ] is perhaps the closest study to our work in terms of scope, focusing on data-centric approaches


to resolving the bias issues at the root cause, i.e., data. However, their scope is much broader in terms of the covered


domains and focuses on identifying current research gaps in data management territory for tackling bias. Besides, they


do not differentiate between different types of bias. This has led to interchangeably using bias and unfairness terms,


while our work solely focuses on representation bias. Moreover, [ 18 ] mostly goes as far as introducing the works at


a high level, while in our work we provide taxonomies and discuss technical details, with running examples where


applicable. Overall, the two surveys are in different abstract levels and have different purposes and contributions.


6


Representation Bias in Data: A Survey on Identification and Resolution Techniques Woodstock ’18, June 03–05, 2018, Woodstock, NY


Stoyanovich et al. [ 152 ] review fairness-related literature in data management pipeline in the context of automated


decision systems’ lifecycle. The article focuses on pre-existing, technical, and emergent bias types and how they are


introduced to the data in different stages of the data management pipeline. Our survey takes a different approach by


focusing on the identification and resolution of representation bias for different types of data, and focuses on the variety


of data-centric techniques for these issues. Therefore, the two works have different scopes. For the same reasons, our


work is distinguished from Catania et al. [34] as it has a similar scope to [152] and follows a very similar outline.


Abiteboul et al. [ 5 ] discuss a few regulatory frameworks, such as the European union’s GDPR, and how the data man

agement community can address challenges such as neutrality, fairness, data protection and transparency highlighted


by these regulations.


Jagadish et al. [ 85 ] is based on the famous white paper written by prominent researchers on the big data lifecycle


and the challenges that are faced in big data analysis. However, they do not cover the responsible dimension of data


analysis in their scope.


Firmani et al. [ 66 ] is a short paper that introduces an ethics cluster and reiterates the challenges in the information


extraction pipeline associated with data quality. Our work falls into the diversity and fairness aspects of the proposed


cluster.


Mehrabi et al. [ 121 ] is a comprehensive survey that classifies different kinds of bias and reviews the body of literature


on machine learning fairness. We would like to highlight that [ 121 ] is a complementary survey to our work that covers


the general topics of bias and fairness (not necessarily data-centric) in breadth and depth. The intersection of [ 121 ] and


our work are only the preprocessing techniques to promote fairness as they do not consider bias measurement methods


such as coverage.


Similarly, Orphanou et al. [ 133 ] reviews the body of works on bias detection, fairness promotion, and explainability


in algorithmic systems from different research communities. Overall, they have a broader scope than our work and for


the same reasons as before, the intersections are only the preprocessing techniques to promote fairness.


Finally, surveys and tutorials such as [ 42, 62 ], focus on identifying and mitigating bias within a specific research


community and/or a specific type of bias and/or specific tasks.


**3** **AN OVERVIEW OF REPRESENTATION BIAS**


With the abundance of data collected from a wide range of contexts, we are transitioning from decision-making based


on intuition and anecdotal observations to decision-making based on the data. Data-driven decision-making has great


potential, and success stories abound. But there are also failures, usually because the larger volume can make it easier


to hide many problems. It is said that every decision is only as good as the data used to make it [ 20 ]. One of the most


important, aspects of data quality is being representative of all the possible subgroups influenced by that decision


[ 66 ]. This representativeness originates from how the data has been collected. With a prospective data collection


approach, such as through a survey or a scientific experiment, data scientists may be able to specify requirements


like representation in data. However, more often than not, data, now known as found data, is collected independently


in a process that data scientists have limited or no control over. Besides, it is important to note that while data must


follow the actual production distribution, this is not sufficient for the development of representative data. The data must


include enough examples from "less popular regions" of data space if these regions are to be handled well by the system.


In today’s data-driven world, Automated Decision Systems (ADS) are widely used in society, ranging from fire


prevention by predicting high-risk buildings to recruiting automation by screening for competitive candidates. However,


historical data used for decision-making might not be objective; it could inherit historical biases in the algorithm


7


Woodstock ’18, June 03–05, 2018, Woodstock, NY Nima Shahbazi, Yin Lin, Abolfazl Asudeh, and H. V. Jagadish


design. For the responsible development of ADS, it is essential to analyze the representation to avoid the potential


risks of injustice. For example, an attempt of the Boston government [ 64 ] using a system to assign students to schools


near their residential areas was found problematic as it ignored the fact that top schools are typically less common


in underprivileged districts. For systems that rely on machine learning algorithms, without a careful inspection


of the training data quality, under-representation of minority groups may cause discrimination in the prediction


results [ 12, 39, 66, 114 ]. For example, StyleGAN [ 98 ], one famous algorithm for auto-generating eerily realistic human


faces, is also producing white faces more frequently than faces of people of color. The problem appears inherited from


the training data sets, which default to white features. As a result, recent research has started to explore the relationship


between machine learning bias and the inadequate sample sizes [ 39, 131 ]. Representation bias is also a crucial problem


in critical domains, such as health care. First of all, there are group-specific patterns in the healthcare data. For example,


many diseases are correlated with demographic factors like race, gender, etc. Ashkenazi Jewish women are known to


have a higher risk of breast cancers [ 59 ]; the likelihood of many diseases, including obesity, hypertension, diabetes, and


high total cholesterol, also varies across racial/ethnic groups [1]. Therefore, the medical datasets’ diversity, especially


demographical diversity, is vital when further using the collected data. Besides, as health data are usually sensitive,


patients’ willingness to share the data might vary [ 48 ]. As a result, ensuring the representativeness of the collected data


is essential to avoid inaccurate or biased results in the downstream usage of the data.


**3.1** **Reasons for Representation Bias**


Bias has been studied in the statistical community for a long time [ 129 ] but social data, increasingly used for policy


decision-making and by social scientists and digital humanities scholars, presents a set of different challenges [ 19, 20, 132 ].


At a high level, bias in social data means certain subpopulations in data are more heavily weighted or represented due to


systematic favoritism. It is a deviation from expectation in data and is recognized as a subtle error that sometimes goes


unnoticed, causing skewed outcomes, low accuracy levels, and analytical errors. These biases are sometimes introduced


to the data due to cognitive biases [ 78, 81 ] in human reporting or flawed data collection or preprocessing. We refer the


reader to [ 76, 132 ] for more information about the general topic of biases in social data, the origins, and various types.


The center point of this survey, representation bias, happens for a variety of reasons with no consensus on an exact


set of grounds. With that in mind, we seek the origins of representation bias in one or more of the following:


_**Historical Bias**_ _._ Historical bias is “the already existing bias due to the socio-technical issues in the world” [ 121 ]. An


example of historical bias can be found in Google’s image search results. Searching for the term “CEO United States”,


the results are dominated by images of male CEOs and show fewer female CEO images. This is because only 8.1% of


Fortune 500 CEOs are women, causing the search results to be biased towards male CEOs. This problem has previously


been shown for a variety of job titles, such as ‘CEO’ in [ 109 ], and Google had alleged to have resolved it. These search


results are indeed reflecting reality. However, whether the search algorithms should mirror this reality or not may


depend on the application and is another issue to consider.


_**Underlying Distribution Skew**_ _._ The underlying distribution that data is collected from may lack an equal ratio or


sufficient representation for all of its subpopulations. In such cases, the underlying distribution is inherently skewed,


and there are no discriminatory motives behind it. For example, according to the US Census Bureau [ 2 ], around 7% of


the US population is of Asian descent while 75% of the population is White. Collecting a uniform sample from the US


society, the Asian community is considered a minority in the outcome sample and naturally less represented. However,


8


Representation Bias in Data: A Survey on Identification and Resolution Techniques Woodstock ’18, June 03–05, 2018, Woodstock, NY


this is a reflection of the underlying distribution which the data has been collected from. This reflection of reality may


lead to discrimination against this subpopulation in some applications.


_**Sampling/selection/self-selection Bias**_ _._ Selection bias is introduced to the data when one fails to ensure proper


randomization in selecting people, groups, or tuples of data for analysis. _Sampling bias_ happens on account of a non

random sampling of a population, causing some (sub) populations to be less likely to be sampled. Note that selection


bias is a cause for sampling bias since having selection bias, the collected samples may not represent a random sampling


of a population. _Self-selection bias_, on the other hand, happens when only a subset of a selection population chooses


to participate in an experiment. This bias occurs when the intention of the participants whether to participate in the


research or not creates abnormal or undesirable conditions. Although selection bias, sampling bias, and self-selection


bias are sometimes used interchangeably, it is important to differentiate between them. Let us clarify this distinction


using an example. Consider a researcher who would like to conduct a survey in Chicago, mailing ballots to selected


respondents. Now if the respondents are only selected from some regions (e.g. near downtown), hence failing to ensure


a random representation of different populations in the city, this is an example of _selection bias_ . Suppose there is no bias


in the selection of respondents. However, only a small portion of the invited respondents decide to take the survey and


mail the forms back. This can cause the _self-selection bias_ . To see how, let us consider the famous example where the


survey question is “Do you like responding to surveys?” with two possible options: 1) Yes, I love responding to surveys


2) No, I toss them in the trash. Now suppose only 10% of the respondents opted to take the survey and the collected


results show 99% favored option 1. The result is indeed invalid as the other 90% who decided not to take the survey


would likely have selected option 2! Now, independent of how the survey was taken, if the collected samples are not


random over Chicago’s population, it is an instance of _sampling bias_ .


**3.2** **Measuring Representation Bias**


In this section, we discuss the measures that have been proposed to evaluate representation bias in data.


_3.2.1_ _Representation Rate._ Representation rate is a metric defined in [ 37 ] to identify representation bias w.r.t. the base


rates. Base rate, also known as “prior probability”, refers to the class probability unconditioned on any observation. In


the existing works [ 102, 150 ], an equal base rate is defined as having an equal number of objects for different subgroups


in the data set. In other words, the objects in the selected set should have an equal chance of belonging to each subgroup.


Consider data set D with _𝑛_ tuples and let _𝑛_ _𝑖_ be the number of tuples belonging to subgroup _𝑖_ . That is, for all possible


subgroups _𝑖, 𝑗_ in D, they are represented if _𝑛_ _𝑖_ = _𝑛_ _𝑗_ .


Next, we present the definition of the representation rate. Consider data set D from discrete domain Ω : = Ω 1 ×

- · · × Ω _𝑑_ = { 0 _,_ 1 } _[𝑑]_ where _𝑑_ is the number of dimensions of the dataset. For a threshold _𝜏_ ∈( 0 _,_ 1 ], data set D following


the distribution _𝑝_ : Ω →[ 0 _,_ 1 ] is said to have representation rate of _𝜏_ with respect to a sensitive attribute _ℓ_ if for all



_𝑧_ _𝑖_ _,𝑧_ _𝑗_ ∈ Ω _ℓ_, we have _[𝑝]_ [[] _𝑍_ _[𝑍]_ = [=] _[𝑧]_ _[𝑖]_ []]




_[𝑖]_

~~_𝑛_~~ _𝑗_ [≥] _[𝜏]_ [. The closer] _[ 𝜏]_ [is to zero, the]



_𝑝_ _[𝑝]_ [ _𝑍_ = [=] _𝑧_ _[𝑧]_ _[𝑖]_ _𝑗_ ] [≥] _[𝜏]_ [. That is, for all possible subgroups] _[ 𝑖, 𝑗]_ [we have] ~~_𝑛_~~ _[𝑛]_ _[𝑖]_ _𝑗_



more biased D is. Representation rate might be hard to achieve. That is because, in practice, it rarely happens that all


subgroups have (almost) the same number of objects.


_3.2.2_ _Data Coverage._ The notion of data coverage has been studied across different settings in [ 6, 8, 12, 13, 88, 114,


124, 157 ] as a metric to measure representation bias. At a high level, coverage is referred to as having enough similar


entries for each object in a data set. For a better understanding, let us go over a definition for the generalized notion of


coverage. Consider a data set D with _𝑛_ tuples, each consisting of _𝑑_ attributes _𝑋_ = { _𝑥_ 1 _,𝑥_ 2 _,_ - · · _,𝑥_ _𝑑_ } . Attribute values


9


Woodstock ’18, June 03–05, 2018, Woodstock, NY Nima Shahbazi, Yin Lin, Abolfazl Asudeh, and H. V. Jagadish


may be non-ordinal categorical (e.g. race ) or continuous-valued (e.g. age ). Ordinal attribute values are normalized to


lie in the range [ 0 _,_ 1 ], with values drawn from the set of rational or real numbers. For every tuple _𝑡_ ∈D, _𝑡_ [ _𝑖_ ] shows the


value of _𝑡_ on attribute _𝑥_ _𝑖_ ∈ _𝑋_ . In practice, the data scientist may be interested in studying coverage over a subset of


attributes, called “ _attributes of interest_ ”. Examples of attributes of interest are _gender_, _race_, _salary_, etc. Subsequently, _𝑋_ is


assumed to be the set of attributes of interest. The data set also contains target attributes _𝑌_ = { _𝑦_ 1 _,_ - · · _,𝑦_ _𝑑_ ′ } that may or


may not be considered for the coverage problem.

Given a query point _𝑞_ ∈[ 0 _,_ 1 ] _[𝑑]_, where _𝑞_ [ _𝑖_ ] shows the value of _𝑞_ with regard to _𝑥_ _𝑖_ ∈ _𝑋_, _𝑞_ is not covered by the


data set D, if there are not “enough” data points in D that are representative of _𝑞_ . In order to generalize the notion


of coverage, let us define G( _𝑞_ ) as the group of tuples that would represent _𝑞_ . For example, suppose _𝑋_ = {gender}


and _𝑞_ has gender=female . Then the set of female individuals represents _𝑞_ . Let G D ( _𝑞_ ) = G( _𝑞_ ) ∩D . That is, G D ( _𝑞_ )


are the set of tuples in D that represent _𝑞_ . Using this notation, coverage of _𝑞_ is defined as the size of G D ( _𝑞_ ) . That is,


_𝑐𝑜𝑣_ ( _𝑞,_ D) = |G D ( _𝑞_ )| . Given a coverage threshold value _𝑘_, _𝑞_ is covered if and only if _𝑐𝑜𝑣_ ( _𝑞,_ D) _> 𝑘_ . The _uncovered_


_region_ in a data set is the collection of tuples that are not covered by it.


It is important to have a high enough coverage for all meaningful sub-populations in data regardless of the data


space to make sure they are adequately represented. We would also like to emphasize the necessity of _human-in-the-_


_loop_ to ignore semantically incorrect sub-populations, e.g. {gender= male, isPregnant = {True}}. Coverage thresholds


are expected as an input to the problem and are supposed to be determined through statistical analyses as they are


application-specific and vary by context. By borrowing the concept from statistics and central limit theorem, the rule of


thumb suggests the number of representatives be around 30 or as [ 153 ] suggests, for each “minority subpopulation” a


minimum of 20 to 50 samples is necessary.


_3.2.3_ _Representation Rate vs. Data Coverage._ Having discussed representation rate and data coverage, let us further


compare these two measures with an example. Consider a data set D with 1000 tuples each having an attribute _{gender}_


with values { male, female }. In order to satisfy representation rate requirements, the male and female groups should


have close counts _relatively to each other_ . For example, using the threshold _𝜏_ =0.8, the ratio of females-males (assuming


that females are the minorities) should be at least 80%. In other words, given that the data set size is 1K, the data set


should contain at least 445 females. On the other hand, data coverage requires a minimum count for each of the groups


_independent from the counts on other groups_ . So, for coverage threshold value _𝑘_ =100, each of the male and female groups


should at least have 100 tuples to be covered. Finally, comparing the two measures, it is evident that the representation


rate provides stronger guarantees of resolving issues w.r.t representation bias in downstream tasks, however, it is more


restrictive and harder to achieve compared to the data coverage. In particular, when the underlying distribution is


skewed (as explained in Section 3.1), it is not possible to both follow the underlying distribution and fully satisfy the


representation rate.


A connection between the fairness measures and representation bias has been made to prove fairness impossibility


theorems. In particular, Kleinberg et al. [ 102 ] prove when there is an unequal base rate in data (i.e., representation rate


is less than one), it is not possible to satisfy different fairness measures. For example, it is not possible to achieve both


Equalized Odds and Predictive Parity at the same time.


**3.3** **Representation Bias Harms**


Before starting the discussion on representation bias identification, we would like to underscore that although represen

tation bias is important, it does not necessarily imply poor and groundless decision-making of the system. For example,


10


Representation Bias in Data: A Survey on Identification and Resolution Techniques Woodstock ’18, June 03–05, 2018, Woodstock, NY



(a) Original data set (b) Under-represented region close to decision boundary



(c) Under-represented region far from the
decision boundary



Fig. 2. Classification task: whether a query point is inside/outside the cat body. Illustration of classifier’s performance for different
under-represented regions.


in a classification setting, having representation bias on continuous attributes in regions far from the _ground-truth_


decision boundary is likely to be immaterial since those points may not contribute to refining the boundary. Similarly, in


a regression setting, in regions of the training data where the fluctuation of the target value is not much, representation


bias is much less crucial than in regions with a higher fluctuation. In general, it is safe to say that representation bias is


problematic in the regions where the model behind the decision system fails to interpolate adequately based on the


current data sample.


To further verify this, let us consider the following experiments, adapted from Asudeh et al. [ 13 ]. First, consider


a binary classification task to label a query point on the x-y plane as belonging to the body of a cat image or the


background (Figure 2a). The training data is generated by randomly sampling from the image, labeling each sample


point as +1 if inside the cat’s body and -1 otherwise. Next, we intentionally remove the sample points in the training


data that belong to the patch highlighted in Figure 2b to make it under-represented. Using the training data and trying


multiple classification models, while the overall performance of the classifiers is high, they all fail to work for the


under-represented region. In particular, while the overall false-negative rate was less than 5%, it was as high as 54% for


the under-represented region. Relying on the training data, the models create the decision boundary by connecting


the two edges of the cat’s body, missing its ear. As a result, the query points that belong to the ear are misclassified as


background, resulting in a high false-negative rate.


Next, we repeat the experiment, but this time, we remove the sample points belonging to the patch shown in Figure


2c. The performance difference of the models between the overall image and the under-represented region is relatively


small (around 4%), and the model performs well for the under-represented region. Looking at the training data, the patch


does not contribute to defining the decision boundary in this specific classification task and, therefore, has minimal


impact on the model’s performance.


11


Woodstock ’18, June 03–05, 2018, Woodstock, NY Nima Shahbazi, Yin Lin, Abolfazl Asudeh, and H. V. Jagadish


**4** **REPRESENTATION BIAS IN STRUCTURED DATA**


Structured data (a.k.a. tabular data) is the most common type of data available in the real world. Databases are built


upon the concept of organizing data in a structured manner to facilitate tasks such as storage, querying, representation,


etc. Representation bias in structured data has extensively been studied, and various techniques for the related problems


have been proposed. This section discusses the literature on identifying and mitigating representation bias in structured


data. For each dimension, we will go through a detailed description of the research works and discuss their novelties.


Figure 3 depicts the taxonomy we propose for the structured data to categorize different techniques based on their


objectives, capabilities, and assumptions.


**Representation Bias in Structured Data**



Identification



Resolution



Discrete Attribute Space



Continuous Attribute Space

[13]



Add More Data



No More Data to Add



Single
Relation

[12, 45, 88, 139]

[14, 16, 136, 142]

[30, 63, 110]



Multiple
Relations

[114]



Query
Rewriting

[6–8]

[125, 150]



Data Labels

and Data Sheets

[69, 123, 124, 154]



Single
Relation



Multiple
Relation



Proper
Signal

[13]



Collection

[12, 15, 157]



Augmentation

[37, 38, 84, 148]



Integration

[126, 127]

[3, 130, 149]



Fig. 3. Classification of techniques on identifying and resolving representation bias in structured data


**4.1** **Running Example**


We use the Adult Income Dataset [ 113 ] to present running examples to better clarify the reviewed techniques. The


Adult Income Data set is used to predict whether individual income exceeds $50K/yr based on the census data.


Consider a projection of the Adult Income Dataset, shown in Figure 4, with six attributes A = { _gender, race, marital-_


_status, age, hours-per-week, years-experience_ }, among which { _gender, race, marital-status_ } are non-ordinal categorical


and { _age, hours-per-week, years-experience_ } are continuous-valued. The data domain for the categorical attributes are


_gender=_ { male, female }, _race=_ { White, Black, Asian, Hispanic }, _marital-status=_ { single, married }. Any attributes in A


can be considered sensitive attributes. The data set also contains binary ground-truth _𝑌_ = { 1 _,_ 0 } representing whether


an individual makes greater than $50K annually or not.


**4.2** **Identification of Representation Bias**


In this section, we study the works focused on identifying representation bias in structured data. Depending on the type


of the attributes of interest, we categorize the techniques into two classes based on whether they target the problem for


_discrete_ (non-ordinal; e.g. race, gender ) or _continuous_ (ordinal; e.g. age ) attributes. The attributes of interest considered


12


Representation Bias in Data: A Survey on Identification and Resolution Techniques Woodstock ’18, June 03–05, 2018, Woodstock, NY

|id|gender|race|marital-status|age|hr/week|yrs-exp|above-50k|
|---|---|---|---|---|---|---|---|
|1|male|white|single|21|40|3|0|
|2|female|white|single|28|38|5|0|
|3|male|white|married|35|45|10|1|
|4|male|black|single|30|40|8|0|
|...|...|...|...|...|...|...|...|



Fig. 4. A toy illustration of the running example (the Adult Income Dataset)


for representation bias often include sensitive attributes (a.k.a. protected attributes) such as race and gender but are


not necessarily limited to them.


_4.2.1_ _Discrete Attribute Space._ Let us begin with cases where attributes for identifying representation bias are categorical.


To better observe representation bias in such cases, let us consider the following example:


Example 1 (Representation bias in discrete attribute space). _Consider the running example data set (Figure 4)_


_described in Section 4.1. Suppose the categorical attributes_ {race, gender, marital-status} _are used for representation_


_bias identification. Each conjunction of attribute-value assignment for a subset of attributes specifies a subgroup such as_


{race=black ∧ gender=female} _. If there are not enough tuples in the data set matching a specific subgroup, it may not be a_


_suitable data set on which to train a system to make a decision for that group._


The existing work has evaluated representation bias in discrete space using the discrete notion of coverage measure


and representation rate. Many critical research fields have targeted the problem of identifying representation bias from


different perspectives. For example, in machine learning it is important to identify under-represented subgroups in


the data used to build the models as they are at a higher risk of experiencing unfairness in downstream data-driven


algorithms [ 12, 88 ]. Another closely related problem in machine learning is model validation by finding problematic


regions in data that the model will perform poorly [45, 136, 142, 157].


Depending on whether the data is single or multiple related, in the following, we will study the techniques for


identifying representation bias in discrete structured data.


_**Single Relation**_ _._ The majority of the existing works focus on studying representation bias for data sets that populate


data in just a _single_ table.


We begin with [ 12 ] that identifies representation bias in discrete space using the discrete notion of coverage measure.


For cases where attributes of interest are non-ordinal categorical, coverage is defined as having “enough” entries in the


data set matching a particular pattern. A _pattern_ is a string that specifies a subgroup (e.g. gender= male ∧ race= white )


that matches possible values over a subset of attributes of interest. Coverage is usually discussed for groups given by


the conjunction of attribute-value assignments. A constant value is considered as the threshold for coverage, meaning


that a minimum number of entries equal to the threshold value should exist from a subpopulation to be covered. In


discrete data sets, there are multiple attributes each having multiple possible values that form a combinatorial number


of possible patterns. Since patterns are the combination of some or all attributes-values, they can have multiple children


and parents. A pattern _𝑃_ 1 is the parent of pattern _𝑃_ 2, if _𝑃_ 1 can be obtained by replacing one of the deterministic elements


in _𝑃_ 2 with X . Deterministic elements in a pattern have a specified value, while non-deterministic elements are indicated


by X . As a simple example, consider a pattern defined over a single binary attribute _gender_ with domain { male, female


}. Pattern _𝑃_ 1 : (gender= X ) is the parent to either of patterns _𝑃_ 2 : (gender= male ) or _𝑃_ 3 : (gender= female ). Equivalently


13


Woodstock ’18, June 03–05, 2018, Woodstock, NY Nima Shahbazi, Yin Lin, Abolfazl Asudeh, and H. V. Jagadish


patterns _𝑃_ 2 and _𝑃_ 3 are the children of _𝑃_ 1 . Depending on the size and skew in data sets, the coverage of patterns could


be different and Asudeh et al. try to identify patterns that do not have sufficient coverage in an efficient way. If a


pattern is uncovered, all of its children are also uncovered. This suggests that uncovered patterns should be identified


in a way that is not dominated by more general ones, for example, if patterns _𝑃_ 1 : (gender= X ) and _𝑃_ 2 : (gender= male )


are both known to be uncovered, _𝑃_ 1 is said to dominate _𝑃_ 2 if _𝑃_ 1 is the parent of _𝑃_ 2 . Uncovered patterns that do not


have uncovered parents are referred to as maximal uncovered patterns (MUPs). Therefore, the problem of identifying


representation bias using the discrete notion of coverage is defined as followed: Given a data set D defined over _𝑑_


attributes with cardinalities _𝑐_, as well as the coverage threshold _𝜏_, try to find all MUPs.


No polynomial time algorithm can guarantee the enumeration of the entire MUPs, however, several algorithms


inspired by set enumeration and the Apriori algorithm for association rule mining are proposed to efficiently address


this problem. In this regard, Asudeh et al. introduce _Pattern Graph_ data structure that exploits the relationship between


patterns to do less work than computing all uncovered patterns by removing the non-maximal ones. The parent-child


relationship between the patterns is represented in a graph that can be used to find better algorithms. _Pattern-Breaker_


starts from the top of the graph where the general patterns are and moves down by breaking each pattern into more


specific ones. If a pattern is uncovered, then all of its descendants are also uncovered and they can not be an MUP,


even if they have a parent that is covered. Therefore, this subgraph of the pattern graph can be pruned. The issue with


_Pattern-Breaker_ is that it explores the covered regions of the pattern graph and for the cases where there are a few


uncovered patterns, it has to explore a large portion of the exponential-size graph. To tackle this, _Pattern-Combiner_


algorithm is proposed that performs a bottom-up traversal of the pattern graph. It uses an observation that the coverage


of a node at the level of the pattern graph can be computed as the sum of the coverage values of its children.


Example 2 (Pattern-Combiner (Asudeh et al. 2019)). _Consider the subgroup race=_ Asian _AND gender=_ female


_in Example 1, this data pattern is in the bottom layer of the Pattern Graph as it contains no unspecified values. It has no_


_children and three parent data patterns: (race=X_ ∧ _gender=_ female _), (race=_ Asian ∧ _gender= X), and (race=X_ ∧ _gender=_


_X). If we find (race=_ Asian ∧ _gender=_ female _) has enough coverage, all its parents are covered. Pattern-Combiner visits the_


_data patterns in the Pattern Graph in a bottom-up manner, and once we find the covered pattern, we can get the coverage of_


_its parents._


The problem with _Pattern-Combiner_ is that it traverses over the uncovered nodes first and therefore, it will not


perform well for the cases in that most of the nodes in the graph are uncovered. In fact, for the cases where most of the


MUPs are placed in the middle of the graph, both _Pattern-Breaker_ and _Pattern-Combiner_ will not be efficient as they


should traverse half of the graph. Therefore, they propose _Deep-Diver_, a search algorithm based on Depth-First-Search


that quickly finds the MUPs, and use them to limit the search space by pruning the nodes both dominating and dominated


by the discovered MUPs.


Jin et al. [ 88 ], design a system on top of the methods and algorithms proposed in [ 12 ] to investigate representation


bias over the intersection of multiple attributes using the notion of coverage.

The next work by Chung et al. [45] proposes SliceFinder [3] as a solution to address a similar problem to identifying


representation bias in data. They try to determine if a model under-performs on some particular parts of data (referred to


as a data _slice_ ) since the overall model performance can fail to reflect that of smaller data slices. A slice is a conjunction


of attribute-value pairs (similar to patterns in [ 12 ]) and is considered problematic if the classification loss function takes


3 Note that, unlike previous works, this work (as well as [ 14, 15, 63, 136, 139, 142 ], explained later) is model-aware. While this assumption may place these
works in the scope of fairness-related literature, due to their data-centric approaches, we include them in our survey.


14


Representation Bias in Data: A Survey on Identification and Resolution Techniques Woodstock ’18, June 03–05, 2018, Woodstock, NY


very different values between the slice and the rest of the data. Enumeration of all possible slices is not practical and


searching for the most under-performing slices can be deceptive since model performance over smaller slices can be


noisy or they may be too small to have a considerable impact on the quality of the model. The goal is to identify the


top- _𝑘_ largest and most problematic slices for which the model does not perform well. Finding the most problematic


slices requires a balance between the significance of the difference in loss and the magnitude of the slice. To do so, the


disparity between the loss of a slice and its counterpart is calculated using a loss function like _logarithmic loss_ such that


the difference is always non-negative (slice has a higher loss than its counterpart). To determine if the difference is


significant, Chung et al. suggest treating each slice as a hypothesis and performing two tests to determine 1) if the


loss disparity is _statistically significant_ (not observed by chance) and 2) whether the _effect size_ of the disparity is large


enough (how problematic the slice is). Therefore, they find a handful of the largest problematic slices, by taking all


problematic slices with an effect size larger than a threshold and ranking them by size (number of entries). In order


to search for problematic slices, Chung et al. propose three algorithms including a baseline. First, they propose the


_Decision Tree Training_ method in which, they train a decision tree to partition examples into slices defined by the tree.


To find the _𝑘_ -problematic slices, they perform a Breadth-First-Search on the decision tree in which slices in each level


are sorted based on an increasing number of literals, decreasing slice size, and decreasing effect size and filtered whether


they are statistically significant and have large enough effect-size. The advantage of using the decision tree approach is


its natural interpretability and the fact that it needs to be expanded a few levels to find the top- _𝑘_ problematic slices.


Conversely, a decision tree is optimized for classification results and may not find all problematic slices. Besides, in


cases of overlapping data slices, the decision tree will find at most one of them. To overcome the aforementioned


problems, they propose the _Lattice Searching_ algorithm, in which slices form a lattice and problematic slices can overlap.


Lattice searching follows the same procedure as the decision tree training algorithm to search for the problematic slices.


Lattice search can be more expensive than the decision tree training approach and cannot address the scalability issue


of searching over the exponential size of data slices therefore, they suggest employing parallelization and sampling


techniques. To better clarify how the lattice search algorithm works, let us look into an example:


Example 3 (Lattice Search (Chung et al. 2019)). _Consider data set_ D _described in section 4.1. For simplicity, suppose_


_that we are interested in top-2 largest slices only w.r.t. to gender and marital-status attributes, and the effect size threshold_


_is_ _𝑇_ _. Initially, priority queue_ _𝑄_ _includes the entire data as a slice. This slice does not have the required effect size and_


_thus is expanded into slices gender=_ male _, gender=_ female _, marital-status=_ single _, and marital-status=_ married _that are_


_inserted in the queue. Next, suppose that gender=_ female _slice has the minimum effect size_ _𝑇_ _and is therefore dequeued and_


_added to the top-2 results. With none of the remaining slices having an effect size_ _𝑇_ _, the largest remaining slice (supposedly_


_marital-status=_ single _) is expanded. Suppose marital-status=_ single ∧ _gender=_ male _has the minimum effect size_ _𝑇_ _, then_


_it is added to the top-2 results and the algorithm stops. Note that marital-status=_ single ∧ _gender=_ female _is already_


_considered as it is a subset of gender=_ female _slice._


Next work, SliceLine [ 142 ], expands on the idea of the previous work [ 45 ] for exact slice enumeration to find real


top- _𝑘_ problematic data slices. This is due to the fact that none of the methods introduced in [ 45 ] are able to find the real


top- _𝑘_ problematic slices and this uncertainty creates trust concerns. Utilizing frequent itemset mining algorithms and


monotonicity for effective pruning, they present a sparse linear algebra implementation of slice enumeration that is


efficient in practice. To do so, a scoring function is devised that linearizes the errors and sizes by involving the ratio


of average slice error to average overall error, and deducting the ratio of overall size to slice size, while weighting


these segments by the user parameter _𝛼_ . Using this scoring function all slices with a score larger than zero are slices


15


Woodstock ’18, June 03–05, 2018, Woodstock, NY Nima Shahbazi, Yin Lin, Abolfazl Asudeh, and H. V. Jagadish


of interest and will be returned in descending order of their score. They also propose upper bounds for the scoring


function based on which the search lattice can be effectively pruned.


Pradhan et al. [ 139 ] propose a related approach to [ 45 ] to identify the patterns in the data that are responsible for


bias from a causal perspective. Using the same notion of pattern as [ 12 ], they use interventions to measure the effect of


patterns in data that significantly promote bias. To do so, they remove a subset of data that is assumed to be the root of


bias and evaluate whether a classifier built on the remaining data is less discriminatory. The bias of each pattern is


evaluated with the _interestingness_ measure. Given a fairness metric F, data set D and pattern _𝑝_, interestingness of

pattern _𝑝_ is defined as [F] _𝑆𝑢𝑝_ [D] [−F] ( _𝑝_ [D] ) [/] _[𝑝]_ where F D is the bias of a classifier trained on D, F D/ _𝑝_ is the bias of the classifier

trained on intervened D (by removing _𝑝_ from D ), and _𝑆𝑢𝑝_ ( _𝑝_ ) is the fraction of data points that satisfy pattern _𝑝_ .


In order to find the top- _𝑘_ patterns causing the most bias, Pradhan et al. utilize a similar bottom-up approach to the


lattice-based search that we saw in [12, 45].


Azzalini et al. [ 14, 16 ] propose yet another related approach to detect representation bias in the data based on


conditional functional dependencies (CFDs). CFDs are conditional dependencies that apply to only a subset of tuples


specified with a condition. They use techniques proposed in [ 33 ] to explore the CFDs and filter out all of the CFDs


that do not have at least one sensitive attribute and target variable or some of the present attributes are not assigned a


constant value. Among the remaining CFDs, they calculate the difference in confidence without and with the sensitive


attribute on the left-hand side. The confidence value indicates how often the CFD has been true. A positive confidence


difference is indicative of bias toward the sensitive group on the left-hand side of the CFD. Finally, they rank the CFDs


w.r.t. multiple criteria of interest such as support-based (number of tuples affected by the bias in CFD), difference-based


(largest impact of the protected attribute on the right-hand-side), and mean-based (balance between the two prior


criteria).


Pastor et al. [ 136 ] propose the notion of _divergence_ to estimate different classification behavior in subgroups compared


to the overall data set. Divergence measures the difference in statistics such as false-positive rate and false-negative rate


between a subgroup and the entire data set. However similar to [ 45 ], to recognize the problematic subgroups, they only


consider the most frequent patterns with a size larger than a threshold and discard smaller subgroups. Once subgroups


with high divergence are recognized, they check whether they are statistically significant or not due to fluctuations


caused by the finite size of the data set. Next using the notion of Shapley value [ 147 ], they investigate which attributes


in each problematic subgroup are contributing the most to the local and global divergence. In this work, Shapley values


measure the contribution of each attribute value to the subgroup divergence. _DivExplorer_ algorithm extracts frequent


subsets of attribute values and estimates their divergence. It begins by accepting a data set D including the ground-truth


values, the prediction results from a model, and a support threshold value. Next, it examines each data point in D to


be a false-positive, false-negative, or otherwise, and the results are mapped into a one-hot-encoding representation.


Next, depending on the frequent pattern mining (FPM) algorithm of choice (using off-the-shelf techniques), for each


step _𝑖_ in FPM, itemsets with the minimum required support are extracted. Next, the cardinality of each itemset w.r.t. to


the outcome function (false-positive, false-negative, or otherwise) is calculated. If the support of the itemset (sum of


the cardinalities divided by the size of D ) is more than the specified support, threshold, the itemset is added to the


list of frequents. Once all of the frequent itemsets are determined, the outcome rate of interest (false-positive rate,


false-negative rate, Accuracy, etc.) is estimated for all frequent itemsets and the divergence of all frequent itemsets as


the difference outcome rate for the itemset _𝐼_ and the entire data set D is computed and returned.


16


Representation Bias in Data: A Survey on Identification and Resolution Techniques Woodstock ’18, June 03–05, 2018, Woodstock, NY


In another related work, Farchi et al. [ 63 ] propose _Shapley Slice Ranking Mechanism with focus on Error concentration_


_(SSR-E)_ as an approach to rank data slices by the order of being problematic. However, they assume the slices are given


as an input and they use the notion of Shapley value to rank the slices. They model the slices as players in a cooperative


game and capture the importance of error concentration and statistical significance of the slices by defining various


characteristic foundations. SSR-E accepts a model and a data set D with _𝑛_ slice as input. For each slice, the algorithm


calculates the set of data points that are misclassified by the input model. The Shapley value of each slice is calculated as


the independent sum of the originality of its data points. The originality of each misclassified data point is proportional


to the number of slices to which it belongs. Finally, the slices are returned in a non-increasing order w.r.t. their Shapley


values.


Cabrera et al. [ 30 ] propose a system called FairVis that employs a different approach to identify underrepresented


subgroups in the combinatorially large space. They perform clustering on the training data set to find statistically similar


subgroups and then use an entropy technique to find important features that are more dominant in that subgroup.


When a feature’s entropy is too close to zero, it means that it is concentrated in one value, which makes the feature


more dominant in that subgroup. Next, they calculate a fairness score on the clusters and present the subgroups to


the user sorted by the score. Once a problematic subgroup has been identified, users can compare them with similar


subgroups to discover which value differences impact performance or to form more general subgroups with fewer


features. The similarity between a pair of subgroups is calculated by summing the Jensen-Shannon divergence between


all features.


Finally, in [ 110 ], Lees et al. suggest exploring each subpopulation’s sample complexity bounds for learning an


approximately fair model with a high probability. Sample complexity provides a lower limit on the count of training


samples that are necessary from the subpopulations to learn a fair model. They demonstrate that a classifier can be


representative of all subgroups if adequate population samples exist and the model dimensionality is aligned with


subgroup population distributions. In case the sampling bias of the subpopulations is not met, human interventions in


the data collection process by correcting representation bias (for example, collecting more data for under-represented


subpopulations) are recommended.


_**Multiple Relations**_ _._ In the real world, data is more commonly stored and integrated into databases with _multiple_


tables. In order to analyze the representation bias, a combinatorial number of attribute-value combinations from different


tables needs to be explored. In this process, the data to be analyzed is obtained through complex operations, e.g., table


joins and predicate combinations, in databases with multiple relations. Due to the sheer data volume, determining


adequate coverage can require a prohibitively long execution time. In [ 114 ], Lin, et al. focus on the threshold-defined


coverage identification in the multiple table scenario. Following the definition of the data pattern and MUP in the


single table scenario, the coverage of a pattern _𝑃_ in a database with multiple relations is defined as the number of


records satisfying _𝑃_ in the equal join result over all the tables. The coverage analysis for multiple relations has two main


challenges: (1) For a given data pattern _𝑃_, to determine its coverage in the database, we need to execute a conjunctive


COUNT query with table joins. It would be hard for the users to enumerate the queries for all data patterns and the


execution time for a combinatorial number of such queries is prohibitive. Query optimization for the set of conjunctive


COUNT queries to determine MUPs is needed for coverage analysis. (2) In the lattice space of the pattern graph, we need


to design search algorithms to identify the set of MUPs with the minimum number of COUNT executions. The authors


design a highly parallel index scheme to handle joins and cross-table predicate combinations to efficiently compute the


number of records for each given group. As discussed in [12], the MUP identification problem is an NP-hard problem.


17


Woodstock ’18, June 03–05, 2018, Woodstock, NY Nima Shahbazi, Yin Lin, Abolfazl Asudeh, and H. V. Jagadish


To traverse the combinatorially large search space of the pattern graph, [ 114 ] designs a priority-based search algorithm


that could minimize the number of computations to assess the count for a given group. The priority-based algorithm


keeps searching the nodes with higher pruning efficiency. When a node is dominated by MUPs or dominates a covered


pattern, it prunes this branch based on the coverage monotonicity property. The priority of the nodes is computed by a


heuristic priority scoring function:


_𝑝𝑟𝑖𝑜𝑟𝑖𝑡𝑦_ = _𝜔_ _𝑝_ × _𝑛_ _𝑝_ + _𝜔_ _𝑐_ × _𝑛_ _𝑐_


where _𝑛_ _𝑝_ and _𝑛_ _𝑐_ are the numbers of parent nodes and child nodes for each data pattern, _𝜔_ _𝑝_ and _𝜔_ _𝑐_ are the weights for


parents and children. With a higher weight for child nodes, the priority algorithm would be close to top-down BFS,


while with a higher weight for parent nodes, the algorithm is more likely to traverse deep to the lower layers.


Besides, as the number of patterns does not need the exact counts for the patterns, we only need to determine


whether the database contains more records than the given threshold or not. Therefore, this paper also provides a


sampling-based approximate algorithm for coverage identification, which allows more efficient computation with


smaller data sizes.


Example 4 (Priority Search Algorithm (Lin et al. 2020)). _Consider the search process in Example 2, the search of_


_the priority-based algorithm will start from the root pattern: (race=X_ ∧ _gender= X), where X represents unspecified values._


_Suppose it is covered and we need to explore its children to find the set of MUPs. Next, we evaluate all its children, suppose_


_among its children, the pattern (race=X_ ∧ _gender=_ female _) has more descendants than the pattern (race=_ Asian ∧ _gender=_


_X) (because of the different cardinalities of race and gender.). The priority-based algorithm will first compute the coverage_


_of the pattern (race=X_ ∧ _gender=_ female _), as once we determine its coverage, we can prune more patterns in the search_


_process._


_4.2.2_ _Continuous Attribute Space._ Data in the real world often consists of a combination of continuous and discrete


values. To better understand representation bias in continuous data sets, let us look further into our running example:


Example 5 (Representation bias in continuous attribute space). _Consider a model trained on data set_ D _described_


_in section 4.1. While the model can discriminate w.r.t. categorical attributes like sex and race, it may also discriminate based_


_on continuous-valued attributes such as age (e.g., because most tech workers and job applicants are young). If there are not_


_enough entries for different age ranges (e.g. age>40) in a data set, it may not be trained with enough data to make a decision_


_for those ranges._


Regarding the example above, simple solutions like binning age into "young" and "old" can transform the continuous


space into discrete. However, they may lead to coarse groupings that are sensitive to the thresholds chosen. It may be


inappropriate to treat a 35-yo as young but a 36-yo as old.


Techniques in this category assume data with continuous-valued attributes and propose solutions for identifying


representation bias in such data sets.


Following a similar definition of coverage discussed earlier in [ 12 ], Asudeh et al. [ 13 ] extend the notion of coverage


to continuous space for identifying representation bias. The problem of identifying representation bias using the


continuous notion of coverage is defined as follows: Given data set D with _𝑛_ tuples over _𝑑_ attributes, and vicinity


radius _𝜌_ and coverage threshold _𝑘_, identify the uncovered region. A query point in continuous data space is covered if


there are enough (at least _𝑘_ ) data points in its _𝜌_ -vicinity neighborhood. _𝜌_ -vicinity neighborhood is the circle centered at


the query point with radius _𝜌_ . The uncovered region is demarcated by the collection of all the uncovered query points


in the space.


18


Representation Bias in Data: A Survey on Identification and Resolution Techniques Woodstock ’18, June 03–05, 2018, Woodstock, NY



Fig. 5. identifying the covered region in the gray
Voronoi cell.



Fig. 6. Covered region in data set D marked in green.
The covered region is the union of all the covered
points in each Voronoi cell.



Depending on the number of attributes in a data set, they propose two algorithms for identifying uncovered regions


in data. First algorithm known as _Uncovered-2D_ studies coverage over two-dimensional data sets where _𝑋_ = { _𝑥_ 1 _,𝑥_ 2 } .


In order to find the number of circles that a query point falls into and consequently discover the uncovered region,


_Uncovered-2D_ makes a connection to _𝑘_ -th order Voronoi diagrams. Consider a data set D and its corresponding _𝑘_ -th


order Voronoi diagram. For every tuple _𝑡_ ∈D, let ◦ _𝑡_ be the _𝑑_ -dimensional sphere ( _𝑑_ -sphere) with radius _𝜌_ centered at _𝑡_ .


Consider a _𝑘_ -voronoi cell V( _𝑆_ ) in the _𝑘_ -th order Voronoi diagram _𝑉_ _𝑘_ (D) . Any point _𝑞_ inside the intersections of the


_𝑑_ -spheres of tuples in _𝑆_, i.e. _𝑞_ ∈∩◦ _𝑡_, is covered, while all other points in the region are uncovered. The algorithm starts
∀ _𝑡_ ∈ _𝑆_

by constructing the _𝑘_ -th order Voronoi diagram of the data set and then for each Voronoi cell V( _𝑆_ ) in the diagram,


it computes the intersection of the circles of the tuples in _𝑆_ and marks the portion of V( _𝑆_ ) that falls outside it as


uncovered. After identifying the uncovered region, a 2D map of { _𝑥_ 1 _,𝑥_ 2 } value combinations is used to report the region


to the user.


Let us look into how _Uncovered-2D_ performs on our running example:


Example 6 (Uncovered-2D (Asudeh et al. 2021)). _Consider data set_ D _described in section 4.1. Suppose that we are_


_interested in identifying the uncovered region w.r.t. to hours-per-week and years-experience attributes. Suppose that a query_


_point is covered if it has two data points in its 0.1 radii. As illustrated in Figure 5, the algorithm generates the 2nd order_


_Voronoi diagram for_ D _, and for each Voronoi cell, the intersection of the two closest circles with radius 0.1 is considered to_


_be the covered region. Figure 6 shows the covered and uncovered regions in_ D _._


The algorithm for the 2D case can be extended to the general case by relaxing the assumption on the number of


attributes to discover the exact uncovered region, however, due to the curse of dimensionality, the search size space


explodes as the number of dimensions increases and as a result, the algorithm will not be practical. Therefore, they


propose a randomized approximation algorithm based on the geometric notion of _𝜀_ -net [ 82 ]. In short, _𝜀_ -net approximates


a set using a collection of simpler subsets. Let X be a set and R be a set of subsets of X . A set N ⊂X is an _𝜀_ -net for X


if for any range _𝑟_ ∈R, if | _𝑟_ ∩ _𝜒_ | _> 𝜀_ | _𝜒_ |, then _𝑟_ contains at least one point of _𝑁_ . The idea is to take random samples


19


Woodstock ’18, June 03–05, 2018, Woodstock, NY Nima Shahbazi, Yin Lin, Abolfazl Asudeh, and H. V. Jagadish


from the space (every sample is a potential query point) and check whether each point is covered or not and label them


as + 1 if uncovered and − 1 otherwise. If we have enough samples in this collection, an _𝜀_ -net is formed using which the


uncovered region can be learned. The problem with _Uncovered-MD_ is that theoretically speaking, in adversarial cases,


the number of samples may be exponentially large to the number of dimensions. However, in practice, the adversarial


case is unlikely to happen since the boundary complexity depends on the number of arcs constructing it which can be


significantly less than the theoretical upper bound provided for the number of samples.


**4.3** **Resolving Representation Bias**


After identifying representation bias in data, the next step is presenting a remedy for it. The first approach to tackling


this problem is adding more data while hoping to address the under-representation issues. However, with limited


control over the data collection processes, it could be difficult and expensive for the data scientist to collect more data


from the data sources. When adding more data is not feasible, the current research suggests preventive solutions such


as informing the user about the representation bias issue or rewriting queries to meet the representation constraints.


With that being said, we would like to emphasize the necessity of human-in-the-loop in the resolution process. It is


vital to notice that not all the under-represented regions in the data are meaningful, and some may even be invalid.


Therefore, a domain expert must evaluate and semantically validate the identified groups/regions.


Generally speaking, resolution techniques that operate by adding more samples to the data set (e.g. data collection,


data integration, etc.) require additional sources of data available that can be employed to resolve representation bias.


Such techniques are effective when representation bias is due to the reasons such as sampling and selection bias. On the


other hand, when reasons such as historical bias cause representation bias, it is costly, if not unlikely, to find additional


sources to collect enough data from minorities. In such cases, preventive techniques (e.g. generating warning signals,


nutritional labels, query rewriting, etc.) are effective to help the users make informed decisions.


In the following, we will introduce state-of-the-art techniques for resolving representation bias in structured data.


_4.3.1_ _Adding More Data._ Enriching the data set with more data is the best way to address the under-representation


issues. However, adding more data is not free. In particular, when the representation bias is due to the underlying


distribution skew (see Section 3.1), collecting more data from the under-represented groups may violate the i.i.d


sample requirement, as the data may no longer follow the underlying distribution. Furthermore, there are not always


opportunities for adding more data through data collection or integration. In these cases, the existing research has


acquired techniques like data augmentation to potentially improve whatever data is available and address the lack of


representation issues.


_**Data Collection**_ _._ Data collection is usually costly. If the data are obtained from some third party, there may be a


direct monetary payment. If the data are directly collected, there may be a data collection cost. In all cases, there is a


cost to cleaning, storing, and indexing the data. To minimize these costs, as little additional data as possible should be


acquired to meet the representation constraints.


In this regard, Asudeh et al. [ 12 ] suggest identifying the smallest number of additional data points needed to hit all


the large uncovered spaces. Given the combinatorial number of patterns, it is not feasible to cover all of the patterns


in practice. To do so, they determine the patterns for the minimum number of items that must be added to the data


set to reach a desired maximum covered level or to cover all patterns with at least a specified minimum value count.


This problem translates to a hitting set instance which can be viewed as a bipartite graph with the value combinations


on the left side and the uncovered patterns on the right. There is an edge between a combination and a pattern if the


20


Representation Bias in Data: A Survey on Identification and Resolution Techniques Woodstock ’18, June 03–05, 2018, Woodstock, NY


combination matches the pattern. The objective is to select the minimum number of nodes on the left side that hit all the


patterns on the right. The hitting set problem is NP-complete and the greedy approach to select the value combination


that hits the maximum number of un-hit patterns guarantees a logarithmic approximation ratio for it.


Example 7 (Coverage Enhancement (Asudeh et al. 2019)). _Consider the example in Section 4.1, consider attributes_


_{race, marital-status, gender}, suppose the set of MUPs contains two patterns:_ _𝑃_ 1 _: (race=X_ ∧ _marital-status =_ Single ∧


_gender=_ female _) and_ _𝑃_ 2 _: (race=_ Asian ∧ _marital-status =X_ ∧ _gender=_ female _). A run of the greedy algorithm picks a_


_pattern (race=_ Asian ∧ _marital-status =_ Single ∧ _gender=_ female _), and this pattern hits both_ _𝑃_ 1 _and_ _𝑃_ 2 _in MUPs, therefore,_


_the coverage enhancement process finishes._


Azzalini et al. [ 15 ] propose an approach to mitigate the representation bias in the data by adding tuples to the CFDs


identified using the techniques in [ 14, 16 ]. There are two ways to add tuples with regard to a CFD. The first option is by


adding tuples to the opposite target variable of the identified CFD. As an example if _(gender =_ female _, marital-status =_


single _)_ −→ _Income = ‘_ ≤ _50K’_ is the identified CFD then tuples should be added to _(gender =_ female _, marital-status =_


single _)_ −→ _income = ‘_ _>_ _50K’_ . The second alternative is adding to advantaged group _(gender =_ male _, marital-status =_


single _)_ −→ _income = ‘_ ≤ _50K’_, however, this method could cause potential issues such as increased discrimination. The


proposed algorithm to optimally add tuples to the data set is an improved version of _Greedy Hit-Count_ algorithm [ 12 ].


For each CFD with the opposite target variable, a vector of size _𝑑_ ( _𝑑_ being the number of attributes in the data set) is


created, and the values of the vector are filled according to the values in the corresponding CFD or _𝑋_ if unspecified.


Next, _Greedy Hit-Count_ algorithm accepts the discovered patterns as inputs and returns the minimum set of tuples


required to repair the data set. After this step, some of the identified CFDs may still be present which can cause bias, so,


in the final step, a correction algorithm removes the tuples associated with the remaining CFDs from the data set.


In another work, Tae et al. [ 157 ] focus on acquiring the right amount of data for data slices such that both accuracy


and fairness are improved. Acquiring the same amount of data for all slices may not have the same cost-benefit and it can


bias the data and affect the model’s accuracy for other regions. Therefore, they propose a few data acquisition strategies


(including 3 baselines) such that the models are accurate and fair for different slices. Baselines include acquiring the


same amount of data for all slices, acquiring data for all slices such that in the end they all have the same amount of


data (Water filling algorithm), and acquiring data in proportion to the original data distribution. None of the baselines


solve the problem in an optimal way and in many cases increase the loss and unfairness of the models. This leads to the


selective data acquisition problem that is defined as given a data set, a set of data slices, a model trained on the data


set, a cost function for data acquisition, and a data acquisition budget, acquire examples for each slice such that the


model’s average loss and average unfairness over all slices are minimized while the overall cost for data collection fits


the budget. The idea is to estimate the learning curves of slices, which reveal the cost benefits of data acquisition. The


impact of data acquisition on the model’s loss is significant at first but then gradually stabilizes to the point where


it is not worth the effort anymore. Given the learning curves, _Slice Tuner_ uses the learning curves to determine how


much data to acquire per slice in order to optimize the model accuracy and fairness across the slices while using a


limited data acquisition budget. However, in reality, learning curves are not perfectly generated because slices may


not have sufficient data for the model loss to be measured. Besides, acquiring data for one slice may affect the loss of


the model on some other slices and eventually change their learning curves. So it is important to generate learning


curves that are reliable enough to still benefit Slice Tuner given these issues. The selective data acquisition problem can


be considered in two different settings: for the cases where slices are independent of each other, it is only needed to


solve the optimization problem once. Since the objective for minimizing loss and unfairness is global, optimization


21


Woodstock ’18, June 03–05, 2018, Woodstock, NY Nima Shahbazi, Yin Lin, Abolfazl Asudeh, and H. V. Jagadish


should be done on all slices. The _One-shot_ algorithm updates the learning curves and solves the optimization problem to


determine the amount of data that needs to be acquired for each slice. When slices are dependent Slice Tuner iteratively


updates the learning curves as more data is acquired. Besides, the iterative updates make the learning curves more


reliable, as they are updated whenever enough influence happens, irrespective of its direction. The _Iterative algorithm_


limits the change of imbalance ratio to determine the amount of data to obtain for each slice. Next, let us look into an


example of how the Iterative algorithm works:


Example 8 (Iterative Algorithm (Tae et al. 2021)). _Recall data set_ D _from section 4.1. Using the aforementioned_


_techniques from [_ _45_ _] slices_ _𝑆_ 1 _of initial size 5 and_ _𝑆_ 2 _of initial size 10 have been identified as the problematic slices. Suppose_


_that the minimum slice size is required to be_ _𝐿_ = 10 _and the data acquisition budget is_ _𝐵_ = 55 _. First, the iterative algorithm_


_acquires 5 tuples for_ _𝑆_ 1 _to meet the required slice size criteria which brings down the budget_ _𝐵_ _to 50 and updates the slice_

_sizes for_ _𝑆_ 1 _and_ _𝑆_ 2 _to_ [ 10 _,_ 10 ] _. Next, the imbalance ratio is calculated as_ [10] ~~10~~ [=] [ 1] _[. While there is still some budget left, suppose]_

_OneShot determines_ [ 10 _,_ 40 ] _tuples to be acquired for_ _𝑆_ 1 _and_ _𝑆_ 2 _. If all of this data is acquired the imbalance ratio will become_

10+40
~~10+10~~ [=] [ 2] _[.]_ [5] _[. Therefore, the difference between the imbalance ratio before and after data acquisition is]_ [ 2] _[.]_ [5] [ −] [1] [ =] [ 1] _[.]_ [5] _[ which]_

_exceeds_ _𝑇_ = 1 _(for simplicity_ _𝑇_ _is a given constant). To avoid exceeding_ _𝑇_ _, change ratio_ _𝑥_ _is calculated such that_ [10] ~~10~~ [+] ~~+~~ [40] ~~10~~ _[𝑥]_ ~~_𝑥_~~ [=] [ 2] _[.]_

_With_ _𝑥_ = 0 _._ 5 _, the number of tuples to be acquired becomes_ 0 _._ 5 × [ 10 _,_ 40 ] = [ 5 _,_ 20 ] _. Next, the data is acquired and budget_ _𝐵_ _,_


_and the rest of the corresponding variables are updated and so long as there is still budget left another iteration of OneShot_


_and the subsequent steps are executed._


_**Data Augmentation.**_ Data augmentation techniques increase the size of data by adding partially altered duplicates


of already existing tuples or generating new synthetic entries from existing data. Some of the existing works adopt these


techniques by adding synthetic points with different values for the attribute of interest for representation. Consequently,


the new data set has an equal number of elements for different values of the attribute of interest, resulting in potentially


resolving the under-representation issues.


In [ 148 ], Sharma et al. propose a novel data augmentation method to address the lack of representation of subgroups


in a data set. For a data set with a protected attribute having a privileged and unprivileged subpopulation, they create


an ideal world data set: for every data sample, a new sample is created that has the same label and features as the


original sample except that it has the opposite value for the sensitive attribute compared to the original sample (e.g.


if the original sample has the sensitive attribute _gender=_ male, the new sample is _gender=_ female and identical to the


original sample w.r.t. the remaining attributes). The synthetic tuples are then sorted in order of their closeness to the


original training distribution and added to the real data set to create intermediate data sets. As a result, this new data


set has an equal number of entries for privileged and unprivileged sub-populations, while the label is not dependent


on the protected attribute anymore, therefore potentially removing representation bias from the model built on the


data set. Although, there is concern about polluting the data set with too many synthetic entries, by selectively adding


the synthetic points that are closest to the original distribution in every increment. The user can see the effect of an


augmentation technique that improves fairness while keeping the overall accuracy nearly constant.


Sometimes, the real-world training data could predominately be composed of majority examples with a small


percentage of outliers or interesting minorities. For example, in applications like fraud detection, disease diagnoses, and


the detection of oil spills, the majority of the records are negative while there is a small number of positive “interesting”


records. Machine learning models trained on such imbalanced data sets are highly likely to have poor performance.


Oversampling is one of the most commonly used methods to enhance the model performance in this case. The naive


uniform oversampling algorithms simply duplicate the minorities uniformly at random and are subject to a higher


22


Representation Bias in Data: A Survey on Identification and Resolution Techniques Woodstock ’18, June 03–05, 2018, Woodstock, NY


risk of model over-fitting. The _Synthetic Minority Oversampling Technique (SMOTE)_ [ 38 ] is a better alternative, which


generates synthetic records of minorities based on their _𝑘_ -Nearest minority neighbors. There is a rich line of works


that extend the SMOTE algorithm, for example, the SMOTE-borderline algorithms [ 77 ], which classified the minorities


into noise, danger, and safe and only uses the danger minorities for data augmentation; and the extension of SMOTE


for high-dimensional data [24].


Similarly, Iosifidis et al. [ 84 ] suggest two techniques for resolving representation bias including an oversampling


baseline by duplicating the instances from the minority subgroups to achieve balance. The idea of their main approach


is to use SMOTE as an augmentation technique. They propose two approaches to creating the instances, first, producing


instances based on a given attribute and populating the minority subgroup for a given attribute. Second, by generating


instances based on a given attribute w.r.t. class, meaning that instances from the under-represented subgroup of a given


attribute are generated to deal with the subgroup’s class imbalance.


Finally, Celis et al. [ 37 ] present a data preprocessing method for mitigating representation bias. The goal of this


approach is to learn a distribution that resolves representation bias while remaining as close as possible to the original


distribution. Learning a distribution in polynomial time to the dimension of the domain (versus domain size that can be


exponential) guarantees the scalability of their method. They propose a framework based on the maximum entropy


principle claiming that of all the distributions satisfying observed constraints, the distribution should be chosen that is


“maximally non-committal” with regard to the current state of knowledge meaning that it makes the fewest assumptions


about the true distribution of the data. Using this principle, probabilistic models of data are learned from samples by


obtaining the distribution over the domain that minimizes the KL-divergence with regards to a “prior” distribution such


that its expectation follows the empirical average derived from the samples. Their approach for preprocessing data


benefits from the maximum entropy framework by combining re-weighting and optimization approaches. Maximum


entropy frameworks can be specified by a prior distribution and a marginal vector, providing a simple way to enforce


constraints for sufficient representation. Using a re-weighting algorithm, Celis et al. specify the prior distribution by


carefully choosing weights for each tuple such that desired fairness measures are satisfied and data is debiased from


representation bias. Let us explain the re-weighting algorithm with an example:


Example 9 (Re-weighting (Celis et al. 2020)). _Consider data set_ D _from section 4.1. Suppose that 3 tuples in_ D _make_


_greater than 50K per year (class positive) and 4 tuples belonging to class make less than the amount (class negative). In the_


_positive class, the gender of 2 of the tuples are_ male _and 1 is_ female _. In the negative class, the gender of 2 of the tuples are_


male _and 2 are_ female _. The weight of each tuple_ _𝑡_ _is calculated as the number of tuples belonging to the class of_ _𝑡_ _divided_


_by the number of tuples belonging to the same class with identical gender as_ _𝑡_ _. Therefore the assigned weights for the tuples_


_in_ D _are calculated as followed:_



_𝑐_ ( _𝑝𝑜𝑠𝑖𝑡𝑖𝑣𝑒_ )
_t=_ female _-positive_ → _𝑤_ ( _𝑡_ ) = _𝑐_ ( _𝑝𝑜𝑠𝑖𝑡𝑖𝑣𝑒,𝑓𝑒𝑚𝑎𝑙𝑒_ ) [=] [3] ~~1~~




[3] _𝑐_ ( _𝑝𝑜𝑠𝑖𝑡𝑖𝑣𝑒_ ) [3]

~~1~~ [=][ 3] _[, t=]_ [ male] _[-positive]_ [ →] _[𝑤]_ [(] _[𝑡]_ [)][ =] _𝑐_ ( _𝑝𝑜𝑠𝑖𝑡𝑖𝑣𝑒,𝑚𝑎𝑙𝑒_ ) [=] ~~2~~



~~2~~ [=][ 1] _[.]_ [5]



_𝑐_ ( _𝑛𝑒𝑔𝑎𝑡𝑖𝑣𝑒_ )
_t=_ female _-negative_ → _𝑤_ ( _𝑡_ ) = _𝑐_ ( _𝑛𝑒𝑔𝑎𝑡𝑖𝑣𝑒,𝑓𝑒𝑚𝑎𝑙𝑒_ ) [=] [4] ~~2~~




[4] _𝑐_ ( _𝑛𝑒𝑔𝑎𝑡𝑖𝑣𝑒_ ) [4]

~~2~~ [=][ 2] _[, t=]_ [ male] _[-positive]_ [ →] _[𝑤]_ [(] _[𝑡]_ [)][ =] _𝑐_ ( _𝑛𝑒𝑔𝑎𝑡𝑖𝑣𝑒,𝑚𝑎𝑙𝑒_ ) [=] ~~2~~



~~2~~ [=][ 2]



Next, a marginal vector is chosen as the weighted average vector of samples to meet the representation rate constraints.


Having defined the optimization program, they solve the dual form using the Ellipsoid algorithm as it can be done in


polynomial time in the dimension of data.


_**Data Integration.**_ In data integration, data is consolidated from different sources into a single, unified view. Thus, it


is a very effective solution to acquire data from different distributions such that sufficient representation is ensured for


the underlying populations. However, there are sampling policy and cost-efficiency concerns that need to be examined.


23


Woodstock ’18, June 03–05, 2018, Woodstock, NY Nima Shahbazi, Yin Lin, Abolfazl Asudeh, and H. V. Jagadish


In this regard, Nargesian et al. [ 126, 127 ] suggest _Data Distribution Tailoring (DT)_ as resolving insufficient repre

sentation of subgroups in a data set by integrating data from multiple sources in the most cost-effective manner such


that subgroups in the data set meet the count distribution specified by the user. Depending on our knowledge about


data source distributions, _DT_ can be defined from two different perspectives, first, when the user is aware of the data


source sizes and the total number of tuples belonging to each subgroup, and second, when such knowledge about the


data sources do not exist. For the cases when the group distributions are known, the process of collecting the target


data set is a sequence of iterative steps, where at every step, the algorithm chooses a data source, queries it, and if the


obtained tuple contributes to one of the groups for which the count requirement is not yet fulfilled, it is kept, otherwise


discarded. To do so, they first propose a _Dynamic Programming (DP)_ algorithm. An optimal source at each iteration


minimizes the sum of its sampling cost plus the expected cost of collecting the remaining required groups, based on


its sampling outcome. The dynamic programming analysis evaluates this cost recursively by considering all future


sampling outcomes and selecting the optimal source in each iteration accordingly.


Example 10 (Dynamic Programming Algorithm (Nargesian et al. 2021)). _Consider the data set schema from_


_Section 4.1. Suppose, to enrich the dataset, one would like to collect more samples from external sources_ D 1 _and_ D 2 _._ D 1 _has_


_20%_ female _and 80%_ male _, and the sampling cost of 2._ D 2 _has 40%_ female _and 60%_ male _, and the sampling cost of 3. For_


_simplicity, suppose that we want to collect one tuple for each demographic group. The DP algorithm calculates the optimal_


_cost 𝐹_ ( _𝑓𝑒𝑚𝑎𝑙𝑒_ = 1 _,𝑚𝑎𝑙𝑒_ = 1) _, and decides the optimal source to query, as follows:_


_𝐹_ (0 _,_ 0) = 0



_𝐹_ (1 _,_ 0) = _𝑚𝑖𝑛_ ( [2]

0 _._

_𝐹_ (0 _,_ 1) = _𝑚𝑖𝑛_ ( [2]



0 _._ 4 [)][ =][ 7] _[.]_ [5][ ⇒] _[query]_ [ D] [2] _[,]_

[3]

0 _._ 6 [)][ =][ 2] _[.]_ [5][ ⇒] _[query]_ [ D] [1] _[,]_




[2] [3]

0 _._ 2 _[,]_ 0 _._

[2] [3]

0 _._ 8 _[,]_ 0 _._



_𝐹_ (1 _,_ 1) = _𝑚𝑖𝑛_ (2 + 0 _._ 2 _𝐹_ (0 _,_ 1) + 0 _._ 8 _𝐹_ (1 _,_ 0) _,_ 3 + 0 _._ 4 _𝐹_ (0 _,_ 1) + 0 _._ 6 _𝐹_ (1 _,_ 0)) = 8 _._ 5 ⇒ _query_ D 1


The drawback to the DP algorithm is that it quickly becomes intractable for cases where the minimum count


requirements for the groups are not small. However, they provide a special case for when the (sensitive) attribute of


interest is binary like _gender (male, female)_ and the cost to query data is similar from all sources. The authors prove


that the optimal selection for this special case is to query the data source with _maximum probability of obtaining a_


_sample from the minority group_ . Similar to the previous algorithm, the process of collecting the target data is a sequence


of iterations where, at every iteration, we should select a data source to query. At each iteration, the algorithm finds


corresponding data sources for each group, and then depending on which group is in the minority, it queries the proper


data source. The algorithm stops when the count requirements of both groups are satisfied and then returns the target


data set. Finally, as an alternative to the DP algorithm, they propose an approximation algorithm for the general case.


They model the problem as _𝑚_ instances of the “coupon collector’s problem”, where every _𝑗_ -th instance aims to collect


samples from the _𝑗_ -th group, and then using the union bound, they come up with an upper-bound on the expected cost


of this algorithm. The algorithm first identifies the minority groups and then queries its corresponding data source and


updates the target data accordingly. Let us look into a simple example:


Example 11 (Coupon Collector’s (Nargesian et al. 2021)). _Consider a case that we desire to collect 100 tuples for_

_group_ G 1 _from the most cost-effective data source for_ G 1 _a.k.a. data source_ D _that has the largest_ _𝑁.𝐶_ _[𝑁]_ [1] _[(]_ _[𝑁]_ [1] _[ is the number of]_

_tuples belonging to_ G 1 _,_ _𝑁_ _is the entire number of tuples in_ D _and_ _𝐶_ _is the sampling cost). Suppose that_ _𝑁_ = 1000 _,_ _𝐶_ = 1

_and 𝑁_ 1 = 200 _, therefore, the cost to collect_ Q 1 = 100 _samples from_ G 1 _is bounded by 𝑁.𝐶._ ln _𝑁_ 1 _𝑁_ −Q 1 1 [≃] [693] _[.]_


24


Representation Bias in Data: A Survey on Identification and Resolution Techniques Woodstock ’18, June 03–05, 2018, Woodstock, NY


For the cases where the group distributions are unknown, Nargesian et al. model DT as a multi-armed bandit problem.


Every data source is an arm and we want to select arms in order to collect the required tuples for each group. Every


arm has an unknown distribution of different groups and a query to an arm has a cost. As the bandit strategy, they


adopt “Upper Confidence Bound (UCB)” to balance exploration and exploitation. At every iteration, for every arm, UCB


computes confidence intervals for the expected reward and selects the arm with the maximum upper bound of reward


to be explored next. Finally, they argue that the reward of obtaining a tuple from a group is proportional to how rare


this group is across different data sources or in other words, what the expected cost one needs to pay is in order to


collect a tuple from that group.


Abernethy et al.[ 3, 4 ] propose an adaptive sampling algorithm that adequately represents sensitive demographic


groups compared to the remaining groups. In each round, the algorithm either samples from the entire population or


the population that is under-represented thus far. The decision to sample from which population depends on a sampling


probability value _𝑝_ which decides whether to minimize the performance loss of the model trained on the current data


( _𝑝_ = 1) or minimize the fairness loss w.r.t. the under-represented group ( _𝑝_ = 0). With that being said, algorithm samples


with a probability of 1 − _𝑝_ from the under-represented group and with a probability of _𝑝_ from the entire population.


Next, the sampled point will be added to the training data, and the algorithm proceeds to the next round.


Shekhar et al. [ 149 ] propose a similar adaptive sampling approach to [ 4 ] based on the optimism principle to actively


create a data set that converges to min-max fair solutions. The optimism principle is used in the multi-armed bandit


literature and tries to identify the hardest group to choose from. Given a fixed amount of budget, the algorithm dedicates


more from the budget to the hardest groups (disadvantaged groups performing worst) w.r.t. a sensitive attribute and


samples more from their distribution.


While [ 4, 149 ] assumes that collecting a fair data set from existing sources is always attainable, this assumption may


not always hold. In this regard, Niss et al. [ 130 ] propose an approach to check the feasibility of collecting a data set


from a set of available sources such that the minority groups are properly represented. To do so, the adaptive sampling


is reduced to the convex hull feasibility problem which is to determine whether a point falls in the convex hull of


the means from a set of unknown distributions. Given a known variable _𝑥_ and a confidence value _𝜖_ _>_ 0 and open set


_𝑥_ _𝜖_ = { _𝑦_ : || _𝑦_ − _𝑥_ || _< 𝜖_ }, a sampling policy is feasible if there exists a _𝑦_ ∈ _𝑥_ _𝜖_ that lies in the convex hull of the means and


otherwise infeasible. They study the convex hull feasibility problem in Bernoulli and Multinomial settings and devise


four sampling algorithms as followed: The _uniform_ algorithm at each iteration chooses from the distribution with the


least samples resulting in a uniform sample size for all distributions. _LUCB Mean_ chooses from the distribution with the


confidence boundary farthest from _𝑥_ in the direction of greatest uncertainty. The direction of greatest uncertainty is


the direction away from _𝑥_, a distribution mean is least likely to lie on. _LUCB Ratio_ chooses from the distribution whose


confidence region has the biggest fraction of area on the side of _𝑥_ in the direction of greatest uncertainty. _Thompson_


_Sampling_ commonly used in the multi-armed bandit literature, samples a mean from the posterior of each distribution,


and chooses the distribution with the mean furthest from _𝑥_ in the direction of greatest uncertainty.


_4.3.2_ _No More Data Available to Add._ It is not always possible to add more data to the data sets as there might be


complications such as unknown underlying distribution, lack of additional data, etc. Existing work suggests alternative


solutions to tackle these scenarios, such as informing the users about the deficiencies in the data set or raising warnings


at query time. Furthermore, by adding proper constraints on the queries w.r.t. the attributes of interest, an effort is


made to ensure the proper representation.


25


Woodstock ’18, June 03–05, 2018, Woodstock, NY Nima Shahbazi, Yin Lin, Abolfazl Asudeh, and H. V. Jagadish


_**Generating Proper Warning Signal.**_ Generating proper signals for the trustworthiness of the analysis [ 13 ] occurs


when querying about a particular data point that might potentially be concerning due to belonging to an under

represented subpopulation. The warning signal states whether the query point is covered or not. For the 2D case, the


idea is to find the Voronoi cell that the query point belongs to and check the point’s distance to all the points from the


data set that fall into that cell. If either of the distances is larger than the vicinity threshold, the query point is uncovered


and a warning signal is generated. For the MD case, the classifier trained on the last iteration of the _Uncovered-MD_


algorithm is used to determine the coverage of the query point by the data set. Finally, whether to consider the outcome


and how to take action is a decision left to the model user.


_**Data Labels and Data Sheets.**_ Annotating data sets with representation information informs the data scientist


about the potential deficiencies due to representation bias when the model is being constructed. This is a signal to


investigate the fitness of data for a particular task before building the models.


In [ 69 ], Gebru et al. propose a list of questions that data set collectors should have in mind before the procedure and


respond to after the collection is done. Users can then make informed decisions about the fitness of the data set for


their tasks. A number of these questions address the representativeness of the data set such as whether the data set


includes all possible instances or is a sample (not necessarily random) of a larger set and if it is the latter, what is the


larger set? Is the sample representative of the larger set and if so how the representativeness was verified, otherwise,


why not? Does the data set identify any subpopulations such as race, gender, age group, etc., and, if so, how are these


subpopulations identified, and what is their distribution like in the data set? Does the data set include attributes that


can be considered sensitive like racial or ethnic origins, sexual orientations, religious beliefs, political opinions, etc?


Some research proposes using data labels to help data users choose the appropriate datasets for their tasks. Information


about data coverage is important to the data set profiling. MithraLabel [ 154 ] provides a set of visual widgets delivering


information about the data set among different tasks on the representativeness of minorities, bias, correctness, coverage


in terms of MUPs, outliers, and much more.


In [ 123, 124 ], Moskovitch et al. design a “coverage label” of compact size that can be used to efficiently estimate


the counts for each combination of discrete attributes (pattern). They provide a trade-off between the label size and


the estimation error of pattern counts. The label model is built upon an estimation function that allows the users to


estimate the count of every pattern. The authors design a label for a given subset _𝑆_ which stores the pattern count


for each possible pattern over _𝑆_ and the value count of each value appearing in the data set. The identification of the


optimal labels is an NP-hard problem. The authors also present an optimized heuristic for optimal label generation.


_**Query Rewriting.**_ Consider a data set with some interesting attributes (for example, gender, race, age) that are


prone to be under-represented and a query over the data. Now suppose that some representation constraints are given


w.r.t. the result of a query when executed over the data set (for example, the number of females to be greater than a


given threshold), but when the query is executed over the data set, results do not satisfy the required constraints. The


idea of query rewriting is to minimally rewrite the transformation queries so that certain representation constraints are


guaranteed to be satisfied in the result of the transformation.


Accinelli et al. [ 8 ] propose an approach for rewriting filter and merge operations in preprocessing pipelines into the


closest operation so that the unprivileged groups are sufficiently represented. This is motivated by the fact that the


under-representation of a subpopulation in an initial or intermediate data set in preprocessing pipelines may lead to the


under-representation of that subpopulation in any future analyses. To do so, they provide an approach that minimally


rewrites the transformation operation such that coverage constraints are ensured to be met in the transformed outcome.


26


Representation Bias in Data: A Survey on Identification and Resolution Techniques Woodstock ’18, June 03–05, 2018, Woodstock, NY


Many potential rewritings could exist, however, their proposed sample-based approximate approach finds minimal


rewriting of the original query. Queries are transformed into a canonical form as a preprocessing step. Next, the


search space of potential rewritings is discretized, in such an order that an approximation of the optimal solution can


be determined in the next step, by inspecting the succeeding finite set of points. The modified input query meeting


coverage constraints can be acquired by examining the grid resulting from the preprocessing step, in an order that


ensures the fast identification of the closest rewriting, and by confirming constraint satisfaction using a sample-based


approach. The coverage-based rewriting is approximate as a result of the discretization of the search space and of


the error in estimating cardinalities and constraint satisfaction on the sample. They propose 3 algorithms including a


baseline for coverage-based query rewriting. Coverage-based Rewriting Baseline _CRBase_ visits the grid in increasing


order of distance from the first cell of the grid. During the visit, we look for the cell corresponding to the query with


the minimum cardinality that satisfies coverage-based constraints. _CRBase with Pruning (CRBaseP)_ adds some pruning


rules to reduce the search space and _CRBase with Pruning with Iteration (CRBasePI)_ further optimizes _CRBaseP_ by


iteratively increasing the number of bins during the search up to a given maximum. As a result, each iteration increases


the precision by which they refine the query and compute the cardinalities. Finally, let us demonstrate how CRBase


algorithm operates on our running example:


Example 12 (CRBase (Accinelli et al. 2020)). _Recall data set_ D _from section 4.1. Consider a simple classification_


_task of whether or not an employee makes greater than 50K a year on individuals working more than 40 hours a week and_


_having more than 5 years of experience. Suppose that the selection conditions hours-per-week>40 and years-experience>5_


_lead to an imbalance in the resulting data set with 130_ single _and 13_ married _individuals while at least 70 of each group is_


_needed. Therefore the query needs to be rewritten such that sufficient coverage for the_ married _group is met. The algorithm_


_initially transforms the selection conditions into canonical form -hours-per-week<-40 and -years-experience<-5. The search_


_space of interest is now -hours-per-week>-40 and -years-experience>-5 and the goal is to find the closest point to_ _𝑄_ (− 5 _,_ − 40 )


_such that the cardinality of_ married _individuals is greater than 70. To do so, the algorithm performs an equi-depth (or_


_equi-width) binning with 4 bins on each dimension in the search space. In the resulting grid, each of the grid points represents_


_an SPJ sensitive query obtained from_ _𝑄_ _by replacing selection constants with the grid point coordinates. Starting from_ _𝑄_ _,_


_the grid points are traversed with various strategies until a point at the minimum distance from_ _𝑄_ _that meets the coverage_


_condition is found. Suppose that this point is_ _𝑄_ [′] (− 4 _._ 5 _,_ − 36 ) _, therefore the query is re-written as hours-per-week>36 and_


_years-experience>4.5._


Since the proposed methods in [ 8 ] are approximate, Accinelli et al. further expand their approach in [ 6 ] by introducing


some measures for computing the appearing errors. These errors include approximation error resulting from the usage


of the grid for the discretization of the query search space, the approximation error correlated with the usage of a


sample during the preprocessing and processing phases, and finally, the error related to the detected optimal rewriting.


As a continuum to the previous works in [ 6, 8 ], Accinelli et al. [ 7 ] further extend the considered queries and


constraints and also the proposed accuracy measures.


Shetiya et al. [ 150 ] propose a fairness-aware query rewriting approach in range queries. They use representation


ratio as their measure of fairness to address selection bias and try to rewrite the original query such that the most


similar results to the original query are returned while meeting the fairness criteria. Depending on the number of


predicates in the query, they propose three algorithms. First, _Single Predicate Query Answering (SPQA)_ algorithm for


single predicate range queries benefits from index jump pointers and quickly looks up fair ranges that have a similarity


27


Woodstock ’18, June 03–05, 2018, Woodstock, NY Nima Shahbazi, Yin Lin, Abolfazl Asudeh, and H. V. Jagadish


of more than a threshold. Jump pointers are linear-size indices that enable sub-linear query answering time. Let us


demonstrate how SPQA works using our running example:


Example 13 (SPQA (Shetiya et al. 2021)). _Consider data set_ D _from Section 4.1. Suppose that we are interested in_


_finding individuals who work greater than 40 hours a week. By performing a selection query on the data set, suppose that_


_we observe a 20% difference between the number of_ male _and_ female _entities in the query outcome. Considering gender_


_equity, we want to have at most a 5% difference between the number of_ male _and_ female _individuals. SPQA finds the most_


_similar fair range to the input query by moving along a jump pointer. Initially, the start end-point of the range is fixed and_


_SPQA expands the end end-point until a fair query is found. When the window indicating the start and end of the fair range_


_is swept to the left, the start end-point can perform a shrink or an expansion. Finally, a fair query (38<hours-per-week<44)_


_most similar to the original query is determined such that the difference in the number of_ male _and_ female _individuals is_


_less than 5% while the Jaccard similarity between the two queries results is_ ≃ _80%._


_Best First Search Multi-Predicate (BFSMP)_ algorithm models the problem of multi-predicate query answering as the


traversal over a graph where nodes represent different queries and there is an edge between two nodes if their outputs


differ by one tuple. Starting from the input range, _BFSMP_ efficiently explores neighboring nodes to find the most similar


fair range. Finally, inspired by the A* algorithm, they propose _Informed BFSMP (IBFSMP)_ which improves _BFSMP_ using


an upper bound on the Jaccard similarity for effective graph exploration.


Moskovitch et al.[ 125 ] propose using the notion of provenance to mitigate bias in databases by finding minimal


query relaxations that increase the number of tuples in groups satisfying a predicate. To do so, the tuples in the data set


are annotated with the query selection conditions, and annotations are propagated in the query evaluation phase. The

annotated provenance value for each tuple is _prov(t) =_ [�] _[𝑖]_ _𝑖_ [=] = _[𝑘]_ 1 _[𝐴]_ _[𝑖]_ [[] _[𝑡.𝐴]_ _𝑖_ []] [ and the provenance inequality of the interested]

constraint is _𝑄_ ( _𝐷_ ) G = [�] _𝑡_ ∈ _𝑄_ ( _𝐷_ ) G _[𝑝𝑟𝑜𝑣]_ [(] _[𝑡]_ [) ≥] _[𝑥]_ [. If the provenance inequality holds for a query then the truth of the]

quality T P ( _𝑝_ ) is _true_ . Next, using the provenance inequality, they present a method for generating minimal relaxations.


They use a minimal changes table (MCT) with values being the terms in the provenance inequality sorted in ascending


order by their minimal change w.r.t. each column. Finally, they traverse the table in a left-right top-down fashion and


keep a result set which they add relaxations or remove them from.


Example 14 (Query Relaxation (Moskovitch et al. 2022)). _Consider the Example in Section 4.1, suppose a query_


_wants to select some people who are aged over 60. The fairness requirement is that the results should contain more than 5_


Black female _aged over 60. However, the query result only gives 3 records satisfying the condition. Query relaxation is_


_used to relax the predicates on the continuous values in the query to include more entities in the result. A minimal relaxation_


_is one that no other query relaxation returns a subset of it. For example, changing the search condition from_ Black female


_aged over 60 to_ Black female _aged over 50 could be a minimal relaxation to get enough records if no other query relaxation_


_on the age attribute is closer to the original query and satisfies the fairness requirement._


**4.4** **Summary**


Finally, in Figure 7, we present an overview of the algorithms/techniques described in this section and present a


side-by-side comparison between them based on different properties. Each technique is associated with its reference


paper and is examined based on the following properties:


  - _Attribute Type_ specifies whether the data is in discrete or continuous space.


  - _Relation Model_ specifies whether data is in single or multiple tables.


28


Representation Bias in Data: A Survey on Identification and Resolution Techniques Woodstock ’18, June 03–05, 2018, Woodstock, NY


  - _Task_ specifies whether the algorithm identifies or resolves insufficient representation.


  - _Technique_ briefly mentions the general idea of the proposed approach.


29


Woodstock ’18, June 03–05, 2018, Woodstock, NY Nima Shahbazi, Yin Lin, Abolfazl Asudeh, and H. V. Jagadish

|Technique|Pruning descendants and/or ancestors of largest uncovered patterns|BFS over the ordered decision tree nodes optimized on the classi i fcation results|BFS over the lattice of all data slices|Using interventions to measure the e f fect of biased patterns|Investigating conditional functional dependencies causing discrimination|Improved exact sparse linear-algebra implementation of slice enumeration algorithm|Labels of limited size to estimate the counts of patterns.|Priority-based algorithm to improve pruning e if fciency of coverage analysis|Exploring large problematic data slices based on divergence|Ranking data slices based on Shapley value|Clustering data set to i fnd problematic subgroups|Using Voronoi diagrams to validate the coverage of a query point 𝑘-th|Creating random samples and learning the uncovered using approximation 𝜀-net|Periodically updating learning curves for dependent slices to learn the amount of data needed to be collected|Transformation to Hitting-Set problem to collect minimum required data|Transformation to Hitting-Set problem to collect minimum required data|Over-sampling minority group instances|Augmenting minority group instances to reach the ideal data set|Re-weighting data to meet representation constraints|Query rewriting such that representation constraints are met|Query rewriting such that representation constraints are met|Minimal query relaxations based on provenance|Exact Dynamic Programming to integrate data from multiple sources|Data integration for binary valued attribute with equal data collection cost|Data integration cost approximation via instances of coupon collector|Data integration modeled as a multi-armed bandit when data source distributions are unknown|Adaptive sampling approach to data integration|Adaptive sampling modeled as a multi-armed bandit for data integration|Reducing adaptive sampling to convex feasibility problem to check the feasibility of data integration|Generating a signal of whether query results can be trusted or not|Generating a signal of whether query results can be trusted or not|Describes data sets from representation perspectives|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Task|Identifcation|Identifcation|Identifcation|Identifcation|Identifcation|Identifcation|Identifcation|Identifcation|Identifcation|Identifcation|Identifcation|Identifcation|Identifcation|Resolution|Resolution|Resolution|Resolution|Resolution|Resolution|Resolution|Resolution|Resolution|Resolution|Resolution|Resolution|Resolution|Resolution|Resolution|Resolution|Resolution|Resolution|Resolution|
|Relation Model<br>Single<br>Multi||||||||✓|||||||||||||||✓|✓|✓|✓|✓|✓|✓|||✓|
|Relation Model<br>Single<br>Multi|✓|✓|✓|✓|✓|✓|✓||✓|✓|✓|✓|✓|✓|✓|✓|✓|✓|✓|✓|✓|✓||||||||✓|✓|✓|
|Attribute Type<br>Discrete<br>Continuous||✓||||||||||✓|✓||||||✓|✓|✓|✓||||||||✓|✓|✓|
|Attribute Type<br>Discrete<br>Continuous|✓|✓|✓|✓|✓|✓|✓|✓|✓|✓|✓||||✓|✓|✓|✓||✓|||✓|✓|✓|✓|✓|✓|✓|||✓|
|Algorithm / Method|Pattern-Breaker, Pattern-Combiner, Deep-Diver [12]|Decision Tree Training [45]|Lattice Searching [45]|Generate Top-k Explanations [139]|FunctionAl dependencIes to discoveR Data Bias [14, 16]|SliceLine Enumeration Algorithm [142]|COUNTATA [124], Pattern Count-based Labels [123]|P-Walk [114]|DivExplorer [136]|Shapley Slice Ranking Mechanism [63]|FairVis [30]|Uncovered-2D [13]|Uncovered-MD [13]|Iterative Algorithm for Slice Tuner [157]|Greedy Coverage Enhancement [12]|Modifed Greedy Hit Count [15]|SMOTE [38]|Greedy Fairness-aware Data Augmentation [148]|Re-weighting [37]|Coverage-based Rewriting Baseline Algorithm with Pruning<br>(with Iteration) [6–8]|Single-predicate range queries answering, (Informed) Best<br>frst search multiple-predicate[150]|Modifed threshold algorithm [125]|Dynamic Programming Algorithm<br>for Data Distribution Tailoring<br>[126]|Equi-cost Binary [126]|Coupon Collector [126]|Upper Confdence Bound [126]|Min-max Stochastic Gradient Descent [3, 4]|Optimistic Sampling for Fair Classifcation [149]|Uniform, Lower Upper Confdence Bound, Thompson Sam-<br>pling [130]|Query-2D [13]|Query-MD [13]|Datasheets for Data sets [69], MithraLabel [154]|



30


Representation Bias in Data: A Survey on Identification and Resolution Techniques Woodstock ’18, June 03–05, 2018, Woodstock, NY


**5** **REPRESENTATION BIAS IN UNSTRUCTURED DATA**


There has been extensive work on techniques for identifying and resolving representation bias in tabular data sets, as


we have discussed above. Additionally, there also is research investigating representation concerns in unstructured data


types such as images, text, and graphs. In this section, we discuss the body of literature on identifying and mitigating


representation bias in unstructured data.


**5.1** **Representation Bias in Image Data**


Computer vision systems have recently achieved outstanding capacity. Identification and resolution of unwanted biases,


specifically the ones due to the disproportionate representation in the image data sets, have drawn a lot of attention


from different research communities. In this section, inspired by Fabbrizzi et al. [ 62 ], we present a taxonomy (as seen in


Figure 8) to classify the techniques and followed by its structure, we review the techniques for debiasing image data


sets. Additionally, while the extent of the works studied in this section is broader than those reviewed in [ 62 ], we would


like to direct the interested reader to [62] for a more comprehensive survey exclusively on the subject.


**Representation Bias in Image Data**



Identification



Resolution



Reduction to

Tabular Data

[28, 55, 122, 164]



Biased Image
Representations

[97]



Cross Data Set

Bias Detection

[101, 145, 158]



Crowd-sourcing

[83]



Data Augmentation

[70, 73, 86, 168]



Reweighting

[112]



Fig. 8. Classification of techniques on identifying and resolving representation bias in image data sets


_5.1.1_ _Identification of Representation Bias._


_**Reduction to tabular data**_ _._ The main idea of this group of techniques is to transform the image data into tabular


data to benefit from the rich literature on the identification of bias in tabular data. The transformation process involves


direct feature extraction from the images using recognition tools and/or indirectly using the metadata of the image


such as description, tag, etc. Of course, these automatic techniques may themselves perpetuate and amplify the biases


in the data as they are prone to errors. In this regard, Dulhanty et al. [55] evaluate two subsets of ImageNet [50] with


human images for representation bias w.r.t. gender and age. They first apply a face recognition algorithm to the data


and next, they apply gender and age recognition models to the outcome. With age and gender attributes determined,


they calculate the distributions among genders and age groups.


Buolamwini et al.[28] created a benchmark data set with balanced entities w.r.t. gender and skin color by counting


and used it to audit the existing gender classification models.


Merler et al. [ 122 ], propose utilizing information theoretical measures of diversity and evenness such as Shannon


entropy, Simpson index, etc. to construct balanced data sets. However, they use existing recognition models or annotators


for labeling the images w.r.t. gender and race.


Wang et al. [ 164 ] build a tool named REVISE for identifying and mitigating bias in visual data sets. Their scope


is limited to three sets of metrics: 1) _Object-based_ that focuses on statistics about object frequency, scale, context, or


31


Woodstock ’18, June 03–05, 2018, Woodstock, NY Nima Shahbazi, Yin Lin, Abolfazl Asudeh, and H. V. Jagadish


diversity of representation 2) _Person-based_ that examines the representation of people from various demographics in


the data set, and allows the user to assess what potential downstream consequences this may have to consider how


best to intervene. It also builds on the object-based analysis by considering how the representation of objects with


people of different demographic groups differs. 3) _Geography-based_ that considers the portrayal of different geographic


regions within the data set and is deeply intertwined with the previous two, as geography influences both the types of


objects that are represented, as well as the different people that are pictured. REVISE accepts annotated image data sets


as input and depending on the annotations it provides insights on the data sets based on each of the three categories


of metrics explained above. Metrics such as object count, scale, co-occurrence, scene diversity, etc. for _Object-based_


category, person prominence, appearance differences, and contextual representations for _Person-based_ and geography


distributions based on people, language, weather and etc. for _Geography-based_ category. REVISE does not claim to find


all the visual biases and it is limited to the available annotations accompanying the data.


_**Biased Image Representations**_ _._ The techniques in this group use distance-based analysis on the low-dimensional


representation of the images in the embedding space to identify representation bias. Particularly, Karkkainen et al. [ 97 ]


create a balanced face data set w.r.t. age, race, and gender. To evaluate the diversity of their data set compared to the


existing work, they visualize the images in 2D using t-SNE [ 159 ], a statistical method for visualizing high dimensional


data by giving each data point a location in 2D/3D space, on the embeddings trained on multiple online sources. Next,


they measure pairwise Manhattan distances between random subsets of the images based on their 128-dimensional


embedding. The skewness of the resulting distribution towards high distances is evidence of high diversity and proper


representation of different subgroups.


_**Cross Data Set Bias Detection**_ _._ Each data set includes specific _signature_ biases that make it distinct from the rest.


This signature bias is introduced in the data collection process and affects the generalizability of the models built on the


data set. This group of methods evaluates the signature bias by comparing different data sets.


In this regard, Torralba et al.[ 158 ] perform some experiments on famous image data sets to measure the bias. To


correctly measure the bias of a data set, it should be compared to the real visual world, which would have to be in the


form of a data set, which could also be biased and, consequently, not a viable option. Therefore, they suggest _Cross-data_


_set Generalization_ by training a model on a data set and testing it on another. Assuming that the training data set is


truly representative of the real world, the model should perform well; otherwise, it means that there are biases, such as


selection and capture, present in the data set. Next, knowing that data sets define a visual phenomenon not only by what


it is but also by what it is not, they argue about _Negative Set Bias_ and whether the negative samples are representative


of the rest of the world or even sufficient. To do so, they run an experiment such that for each data set, a classifier is


trained on its own set of positive and negative instances, and then during testing, the positives come from that data set,


but the negatives come from all data sets combined. The performance of the models shows how well the data set is


representing the rest of the world.


Khosla et al. [ 101 ] propose an algorithm that learns the _visual world_ model and the biases for each data set. The key


observation is that all data sets are sampled from a common _visual world_ (a more general data set). A model trained


on this data set would have the best generalization ability, however, making such a data set is not realistic. Therefore,


they suggest defining the biases associated with each data set and approximating the weights for the visual world by


removing the bias from each data set. The visual world model performs well on average but is not necessarily the best


on any specific data set since it is not biased towards any one data set. On the other hand, the biased model, built by


combining the visual world model and the learned bias, performs superior on the data set that it is biased towards


32


Representation Bias in Data: A Survey on Identification and Resolution Techniques Woodstock ’18, June 03–05, 2018, Woodstock, NY


but does not necessarily extend to the rest of the data sets. In this regard, they propose a maxed-margin learning


discriminative framework to collectively learn the weight vector correlated to the visual world object model and a set of


bias vectors, for each data set such that when combined with the visual world weights lead to an object model specific


to the data set.


Another related work by Schaaf et al. [ 145 ] focuses on measuring bias in image classification tasks by means of


attribution maps. Attribution maps seek to explain image classification models, such as CNNs, by demonstrating the


importance of each individual pixel of the input image on the outcome. To do so, they propose a four-step process to


indicate their usefulness. First, they generate artificial data sets with a known bias. For example, they generate a biased


fruit data set where apples are all on tree backgrounds, while other fruits have different backgrounds, and an unbiased


data set where all fruits have different backgrounds. Next, they train biased CNN models and then generate attribute


maps using different attribution techniques such as _Grad-CAM_, _Score-CAM_, _Integrated Gradients_ and _epsilon-LRP_ . Finally,


they quantitatively evaluate attribution maps’ ability to detect bias using metrics such as _Relevance Mass Accuracy_


_(RMA)_, _Relevance Rank Accuracy (RRA)_ and _Area Over The Perturbation Curve (AOPC)_ . Their results partly confirm the


ability of attribution maps to quantify bias. However, in some cases, attribution maps provide inconsistent results for


different metrics.


_**Crowd-sourcing**_ _._ Hu et al. [ 83 ] propose a crowd-sourcing workflow to facilitate sampling bias discovery in visual


data sets with the help of human-in-the-loop. This workflow takes a visual data set as input and outputs a list of


potential biases of the data set. There are three steps in this workflow. The first step is _Question Generation_ in which the


crowd inspects random samples of images from the input data set and describes their similarity using a question-answer


pair. The next step is _Answer Collection_ in which the crowd reviews separate random samples of images from the input


data set and provides answers to questions generated in the earlier step. Finally, in the third step called _Bias Judgement_,


the crowd judges if the statements about the visual data set automatically generated through the accurate questions


and answers collected in the former steps reflect the real world.


_5.1.2_ _Resolving Representation Bias._


_**Data Augmentation**_ _._ This group of techniques tries to mitigate bias by adding samples for the underrepresented


groups benefiting from the rich literature on image augmentation.


Jaipuria et al. [ 86 ] propose a bias mitigation approach by using targeted synthetic data augmentation that combines


the advantages of gaming engine simulations and _sim2real_ style transfer techniques to bridge the gaps in real data


sets for vision tasks. However, instead of blindly collecting more data or mixing data sets that often end up in worse


final performance, they suggest a smarter approach to augment data regarding the task-specific noise factors. The


results consistently indicate that through adding synthetic data to the training set, a noticeable improvement occurs in


cross-data set generalization, in contrast, to merely training on original data, for a training set of equal size.


Georgopoulos et al. [ 70 ] propose a style transfer approach based on generative adversarial networks (GANs), capable


of creating additional images, reflecting multiple attributes such as race, gender, and age. The resulting data set is less


biased w.r.t. the aforementioned attributes. This is accomplished by relaxing the strict reliance on a single attribute label


and adding a tensor-based mixing structure that multilinearly represents multiplicative interactions between attributes.


Similarly, Yucer et al. [ 168 ] propose another adversarial augmentation method utilizing CycleGANs to transfer race


to mitigate representation bias. They aim to create a synthesized data set by transforming facial images into different


33


Woodstock ’18, June 03–05, 2018, Woodstock, NY Nima Shahbazi, Yin Lin, Abolfazl Asudeh, and H. V. Jagadish


racial domains while maintaining identity-related traits so that race-related traits eventually become irrelevant in


determining the subject’s identity.


Goel et al. [ 73 ] propose an advanced augmentation approach that is oblivious to the differences within subgroups


and aims for class information shared by subgroups. In this regard, they propose CycleGAN Augmented Model Patching


(CAMEL) that first, learns mappings between pairs of subgroups using CycleGANs and creates transformations that


can be used to generate augmented examples based on the training instances and second leverages the transformations


as data augmentations and builds a more robust classifier.


_**Reweighting**_ _._ Li et al. [ 112 ] propose _REPAIR_, a resampling-based bias mitigation approach that is formulated as an


optimization problem. _REPAIR_ assigns a weight to the instances that the classifier built on a feature representation can


penalize more easily. This is implemented through a deep neural network as a feature extractor for the representation


of interest and learning an independent linear classifier to classify the extracted features. Next, bias mitigation is


defined as maximizing the ratio between the loss of the classifier on the reweighted data set and the uncertainty of


the ground-truth labels. Lastly, the problem is reduced to a minimax problem, which can be solved by alternatingly


updating the classifier coefficients and the data set resampling weights, through stochastic gradient descent.


**5.2** **Representation Bias in Natural Language Data**


Natural language processing (NLP) is one of the areas that has widely been affected by the data explosion and


advancement of data-driven decision-making systems. However, the existing biases in the data have regularly resulted


in discriminatory outcomes w.r.t. gender, race, age, disability, etc. Representation bias as one of the key reasons for such


issues has been extensively studied in different NLP tasks such as machine translation, caption generation, sentiment


analysis, hate speech detection, coreference resolution, language models, and word embeddings. Hundreds of technical


papers with a variety of solutions and dozens of reviews have been published tackling different angles of the matter


w.r.t. the task and the target of the bias. Going through, the details of each work is out of the scope of this survey due to


the richness of existing surveys [ 25, 51, 68, 144, 155, 161 ], however, we try to give an overview and a taxonomy of the


techniques (as seen in Figure 9) on identifying and mitigating representation bias in textual data while giving proper


directions to the curious reader.


Representation bias in textual data can happen as a result of the following [155]:


  - _Denigration:_ Using culturally or historically derogatory words.


  - _Stereotyping:_ Heightening the existing societal stereotypes.


  - _Under-representation:_ Disproportionately low representation of a specific group.


Each NLP task can be associated with one or more of these classes as demonstrated in [ 155 ]. Next, inspired by [ 155 ], we


present a taxonomy for the classification of techniques for identifying and mitigating representation bias in textual


data, and following the structure of the taxonomy, we provide a summary of the techniques in the latter sections.


_5.2.1_ _Identification of Representation Bias._ There are two major approaches for identifying representation bias in the


NLP literature:


_**Performance and Representation Difference Among Sensitive Groups**_ _._ Regardless of the task, most NLP model


predictions should not be significantly affected by a sensitive attribute such as gender, race, etc. of the entity. Following


this fact and regarding representation bias in the context of gender, gender swapping and measuring the difference in


evaluation score (such as false-positive rate difference or false-negative rate difference) is a common practice to assess


34


Representation Bias in Data: A Survey on Identification and Resolution Techniques Woodstock ’18, June 03–05, 2018, Woodstock, NY


**Representation Bias in Textual Data**



Identification



Resolution



Performance and Representation
Difference among Sensitive Groups

[17, 52]



Analyzing Sub-space
Embeddings of Sensitive Attribute

[26, 120, 134]



Debiasing Training
Corpora



Debiasing Embeddings



Learning Neutral
Embeddings

[174]



Removing Sub-space
of Sensitive Attribute

[26, 39, 120, 146]



Data Augmentation

[119, 160, 173]



Bias fine-tuning

[135]



Fig. 9. Classification of techniques on identifying and resolving representation bias in textual data


gender bias in such tasks. Furthermore, standard evaluation data sets commonly used in NLP are not sufficient for


measuring gender bias as they often contain bias themselves due to the disproportionate representation of male and


female entities. Therefore, carefully designed task-specific data sets known as Gender Bias Evaluation Test Sets (GBETs)


are constructed that can control the effect of gender bias.


Aside from the performance aspect, Dixon et al. [ 52 ] show how imbalances in the training data w.r.t. representation


can lead to biases in the constructed text classification models with potentially unfair results towards the under

represented group. An example of such biases can be seen in toxicity detection models where due to disproportionate


representation of terms such as “gay” in the training data, statements such as “I’m a gay man” are assigned overly high


toxicity scores even though the comment is not toxic. Models are falsely biased toward words that are disproportionately


represented in toxic comments compared to the overall data set and also they tend to be more biased toward short


comments. To identify the representation bias, Dixon et al. [ 52 ] create a hand-curated list of words for which they


study these two properties. Badjatiya et al. [ 17 ] add two more strategies to what Dixon et al. [ 52 ] proposed to identify


representation bias in textual data. The first strategy is investigating skewed occurrences across classes. If a term


happens to appear in lots of training samples belonging to the toxic class, it encourages the models to classify a comment


containing that particular term as toxic. The second strategy is skewed predicted class probability distribution, which is


the maximum probability of a term belonging to a non-neutral class. A high probability value means that the model has


stereotyped the term to belong to the toxic/non-toxic class.


_**Analyzing Sub-space Embeddings of Sensitive Attribute**_ _._ Word embeddings and language models are trained


on the available biased text corpora and tend to amplify and propagate these biases to the downstream tasks when


used as features. Bolukbasi et al. [ 26 ] investigate representation bias in the context of gender in the embedding space


by showing that geometrically, gender bias can be captured by a direction. Besides, they show that gender-neutral


words (e.g. nurse) are linearly separable from gender-defined words (e.g. queen). Therefore it is possible to differentiate


between the two and capture gender bias in the embedding space. The proposed technique operates as followed: initially,


a set of gender-specific words such as {he, she, man, woman, ...} are chosen as seed words. Using the seed words an


SVM classifier is trained to get the rest of the gender-specific words. The complement of the gender-specific corpus


grants us the set of gender-neutral words. Having the gender-specific and gender-neutral words separated, they select


the seed word pairs such as he-she to act as the x-axis to identify the gender subspace. By checking the distance of


35


Woodstock ’18, June 03–05, 2018, Woodstock, NY Nima Shahbazi, Yin Lin, Abolfazl Asudeh, and H. V. Jagadish


gender-neutral words from the he or she end of the axis (“nurse” closer to she, “genius” closer to he), they identify how


biased the word embeddings are toward such words. These biases originate from the insufficient association of such


words with the opposite gender in the original corpora on which the embeddings were trained.


Manzini et al. [ 120 ] extend this solution to non-binary gender and multi-class sensitive attributes such as race.


religion, etc. Papakyriakopoulos et al.[ 134 ] study detecting representation bias resulting from historical biases reflected


in the word embeddings. To detect the bias of word embeddings, they define an inter-group direction (for instance


between man and woman) and then the bias is quantified as the cosine distance between the word vector and the


inter-group direction. This method compares the magnitude of dependence between a concept and the two groups.


If the concept vector has a higher similarity to a group than another, the concept is considered to be biased in that


direction.


_5.2.2_ _Resolving Representation Bias._ Several methods have been proposed to mitigate representation bias in textual data.


Some of these methods require the models to be retrained after the alterations while some do not and only manipulate


the model to fix the outcomes. In the following we will introduce each of these methods and reiterate some of the


adopted techniques:


_**Text Corpus Alteration**_ _._ To debias the text corpora, two approaches have been proposed:


_Data Augmentation:_ The augmentation approach is to add modified copies of the existing data, or newly created


synthetic data, to the corpora. While some works propose completely removing, masking, or replacing any indication


of gender, race, etc. from the text corpora to eliminate representation bias, De Arteaga et al. [ 49 ] makes an interesting


observation that even by removing the explicit indicators regarding gender, race, or socioeconomic status in the text


corpora, although a slight reduction in representation bias would occur towards the minority group, a significant gap


remains due to the imbalances in the available data between the minority and majority group. Similarly, Li et al. [ 119 ]


make a closely related conclusion for the task of text generation where they investigate representation bias in the


stories generated by GPT-3. They demonstrate how gender stereotypes occur in generated narratives, even in the


absence of gender indicators or stereotype-related cues. They propose prompt design as a possible workaround for


mitigating bias and steering GPT-3, however, they state that it is not a feasible solution for every situation. Zhao et


al. [ 173 ] propose another approach to decrease the bias in text corpora by creating an identical but gender-swapped


version of the original data set and training the model on the union of the original data set, the gender-swapped version


and the named-entity anonymized version of the original data set.


In tasks such as machine translation, due to the domination of male entities in the available text corpora, the


models tend to predict the entities more as male while the actual gender may not be clear. This specifically becomes


problematic while translating into languages such as French where words are gender-specific and masking or removal


of gender indicators is not an option. Vanmassenhove et al. [ 160 ] propose an augmentation technique known as


gender-tagging that tries to solve the aforementioned issue by appending the gender of the entity to the sentences.


Gender-tagging preserves the gender of the speaker and therefore, the machine translation model can consider it while


making predictions.


_Bias Fine-tuning:_ An alternative approach to debias text corpora, proposed by Park et al. [ 135 ], is to use transfer


learning from an already bias-free data set and fine-tune on the biased data to train a model. This approach enables the


models to benefit from bias-free data sets while still sufficiently good to perform the assigned learning task.


36


Representation Bias in Data: A Survey on Identification and Resolution Techniques Woodstock ’18, June 03–05, 2018, Woodstock, NY


_**Word Embedding Adjustment**_ _._ Complete elimination of representation bias from embedding space is not a feasible


goal. However, it has been shown that it is possible to mitigate it w.r.t. the similarity to sensitive attribute subspace and


not needing the embeddings to be retrained. To debias the word embeddings, two approaches have been proposed:


_Removing Sub-space of Sensitive Attribute:_ This is achieved by building a neutral (i.e. genderless, raceless, etc.)


framework for all words [ 146 ] or for gender-neutral words [ 26 ]. For instance, Bolukbasi et al. [ 26 ] propose a neutralization


method to debias the word embeddings. Recall that to identify the bias, they projected each gender-neutral word vector


on an axis with gender-specific words on each end. Having known that the bias exists, they project the gender-neutral


words on the y-axis and thus eliminate the gender bias. Another approach is to make gender-neutral words equidistant


to all words in the gender-specific set meaning that the word “nurse” will be equidistant to sets {he, she} and {man,


woman}. Manzini et al. [120] show that this solution is extendable to non-binary sensitive attributes.


However, Cheng et al. [ 40 ] show that bias w.r.t. different sensitive attributes can be correlated and independent


removal of bias may not be sufficient. To mitigate the bias at a word embedding level, for each bias-sensitive word, they


define a sentiment direction by forming pairs showing different ends of bias (e.g. good-bad, positive-negative, etc.) and


taking the difference between the word embeddings of words in each set and the mean word embedding over the set.


Next, they apply PCA, with the resulting component being the sentiment direction. Next, they define a corresponding


set to the neutral words vector (e.g. doctor, nurse, etc.) and hard-neutralize this vector by making it orthogonal to the


sentiment vector [26].


_Learning Neutral Embeddings:_ Zhao et al. [ 174 ] suggest separating information about the sensitive attribute in a


dimension and keeping the neutral information in other dimensions. In doing so, the sensitive attribute information


can be utilized or neglected on demand. This method requires retraining the embeddings.


_5.2.3_ _Representation Bias in Speech Recognition._ Identification and mitigation of representation bias in speech recogni

tion systems have been briefly studied in the contexts of gender, race, and age [ 65, 103, 117 ]. The primary approach to


identifying the bias in such systems is by measuring the error rate of the speech recognition model among different


subgroups. Demographic information of the speaker is usually acquired through annotations or utilizing automatic


methods [ 41, 61, 141 ]. With the demographic information available, the problem is reduced to bias identification in


tabular data. For the purpose of bias mitigation, diversifying the training data sets w.r.t. race, gender, age, etc. through


the addition of more data is recommended.


**5.3** **Representation Bias in Graphs**


The capacity of graphs to model complex phenomena is gaining increasing attention in many domains, including those


with high societal impact. The sensitivity of applications such as online polarization, job recommendation systems,


disaster response, and criminal justice has led to increasing interest in addressing bias in these systems. There now are


comprehensive studies in the form of review papers and tutorials [ 42, 53, 96 ] to identify biases and promote fairness. In


this section, inspired by Choudhary et al. [ 42 ], we discuss recent techniques to identify and mitigate representation bias


in graphs, present a taxonomy (as seen in Figure 10) of such techniques, and give pointers to the interested reader.


Graphs hold properties such as being non-iid and non-euclidean that make the existing bias identification and


mitigation solutions ineffective. The non-iid assumption suggests that an alteration in one node or edge will affect its


neighbors in the graph. The non-euclidean assumption states that before performing any learning task, a vectorized


37


Woodstock ’18, June 03–05, 2018, Woodstock, NY Nima Shahbazi, Yin Lin, Abolfazl Asudeh, and H. V. Jagadish


**Representation Bias in Graph Data**



Identification



Resolution



Graph-level

[29]



Embedding-level

[27]



Repairing Graph

[106, 151]



Learn Unbiased Embeddings

[29, 100, 140]



Fig. 10. Classification of techniques on identifying and resolving representation bias in graphs


representation of the level of interest (node-level, edge-level, or graph-level) should be learned. Aside from the pre

existing bias in the graphs, different objective functions to learn the representations can perpetuate and amplify the


biases in the graph embeddings. The embeddings should hold two properties:


   - They should reflect the properties of the graph structure.


  - They should be independent of the sensitive attributes.


The first property is guaranteed through the choice of the objective function, however, the second property, is our


problem of interest and can be secured in a two-staged process of identification and mitigation of bias in a variety of


methods.


_5.3.1_ _Identification of Representation Bias._ This set of methods targets representation bias from two different levels:


_**Graph-level**_ _._ Assortative mixing coefficient [ 128 ] is a notion that is used in [ 29 ] to evaluate the homophily of a


graph regarding a particular attribute. This notion is used to evaluate the graph structures for the existing biases. The


values of the assortative mixing coefficient fall into a range of [− 1 _,_ 1 ] and the closer the value to − 1 or 1, the more


correlated the graph is with a sensitive attribute. The mixing coefficient is calculated using the following formula:


_𝑟_ = � _𝑖_ _[𝑒]_ _𝑖_ _𝑗_ [−] [�] _𝑖_ _[𝑎]_ _𝑖_ _[𝑏]_ _𝑖_
1 − ~~[�]~~ _𝑖_ _[𝑎]_ _𝑖_ _[𝑏]_ _𝑖_


where:



_𝑒_ _𝑖𝑗_ = _𝑐𝑎𝑟𝑑_ {( _𝑖, 𝑗_ ) ∈E; _𝐴_ _𝑣_ _𝑖_ = _𝑖,𝐴_ _𝑣_ _𝑗_ = _𝑗_ }



_𝑒_ _𝑖𝑗_ and _𝑏_ _𝑗_ = ∑︁
_𝑗_ _𝑖_



_𝑣_ _𝑖_ _𝑣_ _𝑗_ _, 𝑎_ _𝑖_ =

_𝑚_ ∑︁



_𝑒_ _𝑖𝑗_

_𝑖_



where _𝑣_ _𝑖_ is _𝑖_ -th vertex, E is the set of all edges, _𝑚_ is the number of all edges, _𝐴_ is the sensitive attribute, and _𝑎_ _𝑖_, _𝑏_ _𝑗_ is the


ratio of the edges starting from and ending at each of the attribute values. An _𝑟_ value of zero indicates no bias in the


graph. Mixing coefficient value _𝑟_ can be calculated on any graph to determine bias and promote fairness.


_**Embedding-level**_ _._ Representation Bias (RB) [ 27 ] (should not be mistaken with the topic of our survey though) refers


to the bias in node-level embeddings. RB is calculated using the following:



_𝑅𝐵_ =



_𝑙_
∑︁

_𝑎_ =0



1
| _𝑉_ _𝑎_ | [AUC][({][P] _[ℎ]_ [(] _[𝑎,𝑧]_ _[𝑣]_ [)|∀] _[𝑣]_ [∈] _[𝑉]_ _[𝑎]_ [})]



where _𝑉_ _𝑎_ = { _𝑣_ | _𝐴_ ( _𝑣_ ) = _𝑎_ } is the set of nodes having sensitive attribute value _𝑎_, _ℎ_ is a classifier trained to predict sensitive


attribute _𝐴_ and P _ℎ_ ( _𝑎,𝑧_ _𝑣_ ) is the result of the classification. The idea is to consider the sensitive attribute _𝐴_ as the target


variable and then the aforementioned formula calculates the weighted average of the one-vs-rest AUC values from


38


Representation Bias in Data: A Survey on Identification and Resolution Techniques Woodstock ’18, June 03–05, 2018, Woodstock, NY


the output of the classifier trained to predict _𝐴_ . RB values fall into [ 0 _,_ 1 ] range. The closer the value to 0.5, the more


nondiscriminatory the graph is w.r.t. the sensitive attribute.


_5.3.2_ _Resolving Representation Bias._


_**Repairing Graph**_ _._ The methods introduced in this section try to remove the bias from the graph structure itself


rather than the embeddings. Laclau et al. [ 106 ] try to mitigate the bias in the graph structure using optimal transport


technique in the context of fair edge prediction. They reduce the problem to the problem of alignment between node


distributions of nodes belonging to different sensitive groups based on the rows in the normalized adjacency matrix.


Accordingly, Spinelli et al. [ 151 ] propose a method to modify the adjacency matrix at the training time to balance the


homophily caused by the sensitive attribute. In each training iteration, they remove the edges between nodes based on


a randomized response mechanism between nodes that have the same sensitive attribute value.


_**Learning Unbiased Embeddings**_ _._ The high-level idea of resolving bias for the methods in this section is to place a


fairness constraint on the objective function of the representation learning model. Rahman et al. [ 140 ] try to promote


fairness to Node2vec [ 75 ] by modifying the random walks by changing the transition probabilities to generate unbiased


traces. In consequence, the generated random walk is more likely to have nodes from different groups. Khajehnejad et


al. [ 100 ] propose a re-weighting approach for generating the random walks, however, they assign more weights to the


links that connect nodes from different groups to provide a higher chance of discovery in extreme cases that Rahman


et al. [ 140 ] would have failed. Inspired by Conditional Network Embeddings [ 95 ], Buyl et al. [ 29 ] present a Bayesian


approach that learns debiased representations using as strongly biased as possible prior so that the learned embeddings


have minimal information about sensitive attributes in the training step.


**5.4** **Summary**


In Figure 11, we summarize the papers reviewed on the identification and resolution of representation bias in unstructured


data and present a side-by-side comparison between them based on different properties:


  - _Data Type_ specifies the data type targeted in the corresponding work.


  - _Task_ specifies whether the algorithm identifies or resolves insufficient representation.


  - _Technique_ briefly mentions the general idea of the proposed approach.


39


Woodstock ’18, June 03–05, 2018, Woodstock, NY Nima Shahbazi, Yin Lin, Abolfazl Asudeh, and H. V. Jagadish

|Technique|Reduction to tabular data using metadata and recognition tools to extract attributes of interest|Reduction to tabular data by annotation and counting subpopulations|Reduction to tabular data by annotation and using recognition tools to extract attributes of interest|Reduction to tabular data using annotated data to extract attributes of interest|Distance-based analysis of the image representations in the embedding space|Cross data set generalization by training model on a data set and testing on another one|Learning and removing the biases in di f ferent data sets to approximate the weights for an unbiased visual world|Identifying the bias using the attribution maps|Crowd-sourcing approach to identify representation bias|Data augmentation using targeted synthetic data|Data augmentation w.r.t. di f ferent attributes using style transfer GANs|Data augmentation w.r.t. race attribute using cycleGANs|Data augmentation using cycleGANs|Assigning weights instances in the data that are penalized more easily by the models|Gender swapping and measuring the di f ference in evaluation score, Investigating the e f fect of disproportionate representation in the training data|Investigating the e f fect of disproportionate representation in the training data Investigating skewed occurrences across classes, Investigating skewed predicted class probability distribution|Checking the magnitude of the dependence of a concept and two genders in the embedding space|Checking the magnitude of the dependence of a concept and multiple groups in the embedding space|Checking the magnitude of the dependence of a concept and two groups in the embedding space|Prompt design to reduce gender bias in text generation|Augmenting corpora by appending the gender-swapped version of the text|Augmenting corpora by gender-tagging|Transfer learning from an already bias-free data set and i fne-tune on the biased data to train a model|Eliminating the gender-pair associations from gender-neutral words by making it orthogonal to the gender vector in the embedding space|Eliminating the non-binary gender/race pair associations from gender/race neutral words by making it orthogonal to the gender/race vector in the embedding space|Joint bias removal w.r.t. to di f ferent sensitive attributes by neutralizing the word vectors in the embedding space|Separating information about the sensitive attribute by keeping it in another dimension|Measuring error rate whiting subgroups identi i fed through annotation or automatic recognition tools|Using mixing coe if fcient to evaluate the homophily of a graph w.r.t. a speci i fc attribute|Identifying bias in node embedding level using the notion of Representation Bias|Repair graph by aligning node distributions of nodes belonging to di f ferent sensitive groups|Modifying the adjacency matrix at the training time to balance the homophily caused by the sensitive attribute|Bayesian approach to learn debiased representations using a strongly biased prior|Re-weighting approach by assigning more weights to the links that connect nodes from di f ferent groups for generating random walks|Modifying the random walks through changing the transition probabilities to generate unbiased traces|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Task|Identifcation|Identifcation|Identifcation|Identifcation|Identifcation|Identifcation|Identifcation|Identifcation|Identifcation|Resolution|Resolution|Resolution|Resolution|Resolution|Identifcation|Identifcation|Identifcation|Identifcation|Identifcation|Resolution|Resolution|Resolution|Resolution|Resolution|Resolution|Resolution|Resolution|Identifcation|Identifcation|Identifcation|Resolution|Resolution|Resolution|Resolution|Resolution|
|Data Type|Image|Image|Image|Image|Image|Image|Image|Image|Image|Image|Image|Image|Image|Image|Text|Text|Text|Text|Text|Text|Text|Text|Text|Text|Text|Text|Text|Speech|Graph|Graph|Graph|Graph|Graph|Graph|Graph|
|Paper / System|Auditing ImageNet: towards a model-driven framework for annotating demo-<br>graphic attributes of large-scale image datasets [55]|Gender shades: Intersectional accuracy disparities in commercial gender classif-<br>cation [28]|Diversity in faces [122]|REVISE [164]|FairFace [97]|Unbiased look at data setbias [158]|Undoing the damage of data setbias [101]|Towards measuring bias in image classifcation [145]|Crowdsourcing detection of sampling biases in image datasets [83]|Defating data setbias using synthetic data augmentation [86]|Mitigating demographic bias in facial datasets with style-based multi-attribute<br>transfer [70]|Exploring racial bias within face Recognition via per-subject adversarially-enabled<br>data augmentation [168]|CAMEL[73]|REPAIR[112]|Measuring and mitigating unintended bias in text classifcation [52]|Stereotypical bias removal for hate speech detection task using knowledge-based<br>generalizations [17]|Man is to computer programmer as woman is to homemaker? Debiasing word<br>embeddings [26]|Black is to criminal as caucasian is to police: Detecting and removing multiclass<br>bias in word embeddings [120]|Bias in word embeddings [134]|Gender and representation bias in GPT-3 generated stories [119]|Learning gender-neutral word embeddings [173]|Getting gender right in neural machine translation [160]|Reducing gender bias in abusive language detection [135]|Man is to computer programmer as woman is to homemaker? Debiasing word<br>embeddings [26]|Black is to criminal as caucasian is to police: Detecting and removing multiclass<br>bias in word embeddings [120]|Toward understanding bias correlations for mitigation in NLP [40]|Learning gender-neutral word embeddings [174]|Quantifying bias in automatic speech recognition [65], Racial disparities in auto-<br>mated speech recognition [103], Towards Measuring Fairness in speech recogni-<br>tion: casual conversations data set transcriptions [117]|Debayes: a bayesian method for debiasing network embeddings [29]|Compositional fairness constraints for graph embeddings [27]|All of the fairness for edge prediction with optimal transport [106]|Fairdrop: Biased edge dropout for enhancing fairness in graph representation<br>learning [151]|Debayes: a bayesian method for debiasing network embeddings [29]|CrossWalk: fairness-enhanced node representation learning [100]|Fairwalk: Towards fair graph embedding [140]|



40


Representation Bias in Data: A Survey on Identification and Resolution Techniques Woodstock ’18, June 03–05, 2018, Woodstock, NY


**6** **CONCLUSION**


In this paper, we surveyed techniques for the identification and resolution of representation bias in data. After reviewing


the fairness literature at a high level, we provided a thorough overview of the problem definition, the causes, and how


to measure and quantify this phenomenon in both structured and unstructured data. Depending on the data type, we


then presented taxonomies based on multiple dimensions and had side-by-side comparisons of the techniques. We


discussed the details of several algorithms to illustrate the different challenges and the problems they address. Two


promising research directions we envision being important are:


  - _Addressing representation bias in other types of data sets._ As we discussed in section 5, with the extension of the


problem scope to new data types such as streaming data, spatio-temporal data, etc., new challenges arise and the


current solutions may not be directly extendable.


  - _More metrics for measuring representation bias._ Existing works have introduced _coverage_ and _representation rate_


for measuring representation bias. However, each metric has potential shortcomings that provide new research


opportunities. Furthermore, when it comes to data quality and trust measures in data, there is no such thing as


“enough” and there is always room for improvement.


**ACKNOWLEDGEMENTS**


This research was supported in part by the National Science Foundation, under grants 2107290, 1741022, 1934565, and


2106176.


**REFERENCES**


[1] [n.d.]. Health, United States Spotlight Racial and Ethnic Disparities in Heart Disease. [https://www.cdc.gov/nchs/hus/spotlight/](https://www.cdc.gov/nchs/hus/spotlight/HeartDiseaseSpotlight_2019_0404.pdf )

[HeartDiseaseSpotlight_2019_0404.pdf.](https://www.cdc.gov/nchs/hus/spotlight/HeartDiseaseSpotlight_2019_0404.pdf )

[2] 2019. _The Asian and Pacific Islander Population in the United States: May 2019_ . US Census Bureau.

[3] Jacob Abernethy, Pranjal Awasthi, Matthäus Kleindessner, Jamie Morgenstern, Chris Russell, and Jie Zhang. 2020. Active sampling for min-max

fairness. _arXiv preprint arXiv:2006.06879_ (2020).

[4] Jacob Abernethy, Pranjal Awasthi, Matthäus Kleindessner, Jamie Morgenstern, and Jie Zhang. 2020. Adaptive sampling to reduce disparate

performance. _arXiv e-prints_ (2020), arXiv–2006.

[5] Serge Abiteboul and Julia Stoyanovich. 2019. Transparency, fairness, data protection, neutrality: Data management challenges in the face of new

regulation. _Journal of Data and Information Quality (JDIQ)_ 11, 3 (2019), 1–9.

[6] Chiara Accinelli, Barbara Catania, Giovanna Guerrini, and Simone Minisi. 2021. The impact of rewriting on coverage constraint satisfaction.. In

_EDBT/ICDT Workshops_ .

[7] Chiara Accinelli, Barbara Catania, Giovanna Guerrini, and Simone Minisi. 2022. A Coverage-based Approach to Nondiscrimination-aware Data

Transformation. _ACM Journal of Data and Information Quality (JDIQ)_ (2022).

[8] Chiara Accinelli, Simone Minisi, and Barbara Catania. 2020. Coverage-based Rewriting for Data Preparation. In _EDBT/ICDT Workshops_ .

[9] Tameem Adel, Isabel Valera, Zoubin Ghahramani, and Adrian Weller. 2019. One-network adversarial fairness. In _Proceedings of the AAAI Conference_

_on Artificial Intelligence_, Vol. 33. 2412–2420.

[10] Alekh Agarwal, Alina Beygelzimer, Miroslav Dudík, John Langford, and Hanna Wallach. 2018. A reductions approach to fair classification. In

_International Conference on Machine Learning_ . PMLR, 60–69.

[11] Abolfazl Asudeh and H. V. Jagadish. 2020. Fairly evaluating and scoring items in a data set. _PVLDB_ 13, 12 (2020), 3445–3448.

[12] Abolfazl Asudeh, Zhongjun Jin, and HV Jagadish. 2019. Assessing and remedying coverage for a given dataset. In _2019 IEEE 35th International_

_Conference on Data Engineering (ICDE)_ . IEEE, 554–565.

[13] Abolfazl Asudeh, Nima Shahbazi, Zhongjun Jin, and H. V. Jagadish. 2021. Identifying Insufficient Data Coverage for Ordinal Continuous-Valued

Attributes. In _SIGMOD_ . ACM.

[14] Fabio Azzalini, Chiara Criscuolo, and Letizia Tanca. 2021. FAIR-DB: FunctionAl DependencIes to discoveR Data Bias.. In _EDBT/ICDT Workshops_ .

[15] Fabio Azzalini, Chiara Criscuolo, and Letizia Tanca. 2021. Functional Dependencies to Mitigate Data Bias. In _Proceedings of the 30th Italian_

_Symposium on Advanced Database Systems_ .

[16] Fabio Azzalini, Chiara Criscuolo, and Letizia Tanca. 2021. A short account of FAIR-DB: A system to discover Data Bias. In _29th Italian Symposium_

_on Advanced Database Systems, SEBD 2021_, Vol. 2994. CEUR-WS, 1–8.


41


Woodstock ’18, June 03–05, 2018, Woodstock, NY Nima Shahbazi, Yin Lin, Abolfazl Asudeh, and H. V. Jagadish


[17] Pinkesh Badjatiya, Manish Gupta, and Vasudeva Varma. 2019. Stereotypical bias removal for hate speech detection task using knowledge-based

generalizations. In _The World Wide Web Conference_ . 49–59.

[18] Agathe Balayn, Christoph Lofi, and Geert-Jan Houben. 2021. Managing bias and unfairness in data for decision support: a survey of machine

learning and data engineering approaches to identify and mitigate bias and unfairness within data management and analytics systems. _The VLDB_

_Journal_ 30, 5 (2021), 739–768.

[19] Solon Barocas, Moritz Hardt, and Arvind Narayanan. 2019. Fairness and machine learning: Limitations and opportunities. fairmlbook.org.

[20] Solon Barocas and Andrew D Selbst. 2016. Big data’s disparate impact. _Calif. L. Rev._ 104 (2016), 671.

[21] Yahav Bechavod and Katrina Ligett. 2017. Penalizing unfairness in binary classification. _arXiv preprint arXiv:1707.00044_ (2017).

[22] Alex Beutel, Jilin Chen, Tulsee Doshi, Hai Qian, Allison Woodruff, Christine Luu, Pierre Kreitmann, Jonathan Bischof, and Ed H Chi. 2019. Putting

fairness principles into practice: Challenges, metrics, and improvements. In _Proceedings of the 2019 AAAI/ACM Conference on AI, Ethics, and Society_ .

453–459.

[23] Alex Beutel, Jilin Chen, Zhe Zhao, and Ed H Chi. 2017. Data decisions and theoretical implications when adversarially learning fair representations.

_arXiv preprint arXiv:1707.00075_ (2017).

[24] Rok Blagus and Lara Lusa. 2013. SMOTE for high-dimensional class-imbalanced data. _BMC Bioinform._ [14 (2013), 106. https://doi.org/10.1186/1471-](https://doi.org/10.1186/1471-2105-14-106)

[2105-14-106](https://doi.org/10.1186/1471-2105-14-106)

[25] Su Lin Blodgett, Solon Barocas, Hal Daumé III, and Hanna Wallach. 2020. Language (technology) is power: A critical survey of" bias" in nlp. _arXiv_

_preprint arXiv:2005.14050_ (2020).

[26] Tolga Bolukbasi, Kai-Wei Chang, James Y Zou, Venkatesh Saligrama, and Adam T Kalai. 2016. Man is to computer programmer as woman is to

homemaker? debiasing word embeddings. _Advances in neural information processing systems_ 29 (2016).

[27] Avishek Bose and William Hamilton. 2019. Compositional fairness constraints for graph embeddings. In _International Conference on Machine_

_Learning_ . PMLR, 715–724.

[28] Joy Buolamwini and Timnit Gebru. 2018. Gender shades: Intersectional accuracy disparities in commercial gender classification. In _Conference on_

_fairness, accountability and transparency_ . PMLR, 77–91.

[29] Maarten Buyl and Tijl De Bie. 2020. Debayes: a bayesian method for debiasing network embeddings. In _International Conference on Machine_

_Learning_ . PMLR, 1220–1229.

[30] Ángel Alexander Cabrera, Will Epperson, Fred Hohman, Minsuk Kahng, Jamie Morgenstern, and Duen Horng Chau. 2019. FairVis: Visual analytics

for discovering intersectional bias in machine learning. In _2019 IEEE Conference on Visual Analytics Science and Technology (VAST)_ . IEEE, 46–56.

[31] Toon Calders, Faisal Kamiran, and Mykola Pechenizkiy. 2009. Building Classifiers with Independency Constraints. _2009 IEEE International_

_Conference on Data Mining Workshops_ (2009), 13–18.

[32] Toon Calders and Sicco Verwer. 2010. Three naive Bayes approaches for discrimination-free classification. _Data mining and knowledge discovery_ 21,

2 (2010), 277–292.

[33] Loredana Caruccio, Vincenzo Deufemia, and Giuseppe Polese. 2015. Relaxed functional dependencies—a survey of approaches. _IEEE Transactions_

_on Knowledge and Data Engineering_ 28, 1 (2015), 147–165.

[34] Barbara Catania, Giovanna Guerrini, and Chiara Accinelli. 2022. Fairness & friends in the data science era. _AI & SOCIETY_ (2022), 1–11.

[35] Simon Caton and Christian Haas. 2020. Fairness in machine learning: A survey. _arXiv preprint arXiv:2010.04053_ (2020).

[36] L Elisa Celis and Vijay Keswani. 2019. Improved adversarial learning for fair classification. _arXiv preprint arXiv:1901.10443_ (2019).

[37] L Elisa Celis, Vijay Keswani, and Nisheeth Vishnoi. 2020. Data preprocessing to mitigate bias: A maximum entropy based approach. In _International_

_Conference on Machine Learning_ . PMLR, 1349–1359.

[38] Nitesh V. Chawla, Kevin W. Bowyer, Lawrence O. Hall, and W. Philip Kegelmeyer. 2002. SMOTE: Synthetic Minority Over-sampling Technique. _J._

_Artif. Intell. Res._ [16 (2002), 321–357. https://doi.org/10.1613/jair.953](https://doi.org/10.1613/jair.953)

[39] Irene Chen, Fredrik D Johansson, and David Sontag. 2018. Why is my classifier discriminatory? _arXiv preprint arXiv:1805.12002_ (2018).

[40] Lu Cheng, Suyu Ge, and Huan Liu. 2022. Toward Understanding Bias Correlations for Mitigation in NLP. _arXiv preprint arXiv:2205.12391_ (2022).

[41] Donald G Childers and Ke Wu. 1991. Gender recognition from speech. Part II: Fine analysis. _The Journal of the Acoustical society of America_ 90, 4

(1991), 1841–1856.

[42] Manvi Choudhary, Charlotte Laclau, and Christine Largeron. 2022. A Survey on Fairness for Machine Learning on Graphs. _arXiv preprint_

_arXiv:2205.05396_ (2022).

[43] Alexandra Chouldechova. 2017. Fair prediction with disparate impact: A study of bias in recidivism prediction instruments. _Big data_ 5 2 (2017),

153–163.

[44] Matt J Kusner Joshua Loftus Chris. [n.d.]. Russell and Ricardo Silva. 2017. Counterfactual fairness. _Advances in neural information processing_

_systems_ ([n. d.]), 4066–4076.

[45] Yeounoh Chung, Tim Kraska, Neoklis Polyzotis, Ki Hyun Tae, and Steven Euijong Whang. 2019. Slice finder: Automated data slicing for model

validation. In _2019 IEEE 35th International Conference on Data Engineering (ICDE)_ . IEEE, 1550–1553.

[46] Sam Corbett-Davies, Emma Pierson, Avi Feller, Sharad Goel, and Aziz Z Huq. 2017. Algorithmic Decision Making and the Cost of Fairness.

_Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining_ (2017).

[47] Brian d’Alessandro, Cathy O’Neil, and Tom LaGatta. 2019. A Data Scientist’s Guide to Discrimination-Aware Classification Authors:.


42


Representation Bias in Data: A Survey on Identification and Resolution Techniques Woodstock ’18, June 03–05, 2018, Woodstock, NY


[48] Sabyasachi Dash, Sushil Kumar Shakyawar, Mohit Sharma, and Sandeep Kaushik. 2019. Big data in healthcare: management, analysis and future

prospects. _Journal of Big Data_ 6, 1 (2019), 1–25.

[49] Maria De-Arteaga, Alexey Romanov, Hanna Wallach, Jennifer Chayes, Christian Borgs, Alexandra Chouldechova, Sahin Geyik, Krishnaram

Kenthapadi, and Adam Tauman Kalai. 2019. Bias in bios: A case study of semantic representation bias in a high-stakes setting. In _proceedings of the_

_Conference on Fairness, Accountability, and Transparency_ . 120–128.

[50] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. 2009. Imagenet: A large-scale hierarchical image database. In _2009 IEEE_

_conference on computer vision and pattern recognition_ . Ieee, 248–255.

[51] Mark Díaz, Isaac Johnson, Amanda Lazar, Anne Marie Piper, and Darren Gergle. 2018. Addressing age-related bias in sentiment analysis. In

_Proceedings of the 2018 chi conference on human factors in computing systems_ . 1–14.

[52] Lucas Dixon, John Li, Jeffrey Sorensen, Nithum Thain, and Lucy Vasserman. 2018. Measuring and mitigating unintended bias in text classification.

In _Proceedings of the 2018 AAAI/ACM Conference on AI, Ethics, and Society_ . 67–73.

[53] Yushun Dong, Jing Ma, Chen Chen, and Jundong Li. 2022. Fairness in Graph Mining: A Survey. _arXiv preprint arXiv:2204.09888_ (2022).

[54] Flávio du Pin Calmon, Dennis Wei, Karthikeyan Natesan Ramamurthy, and Kush R. Varshney. 2017. Optimized Data Pre-Processing for Discrimina
tion Prevention. _ArXiv_ abs/1704.03354 (2017).

[55] Chris Dulhanty and Alexander Wong. 2019. Auditing imagenet: Towards a model-driven framework for annotating demographic attributes of

large-scale image datasets. _arXiv preprint arXiv:1905.01347_ (2019).

[56] Cynthia Dwork, Moritz Hardt, Toniann Pitassi, Omer Reingold, and Richard S. Zemel. 2012. Fairness through awareness. _ArXiv_ abs/1104.3913

(2012).

[57] Jeff Edmonds, Jarek Gryz, Dongming Liang, and Renée J. Miller. 2003. Mining for empty spaces in large data sets. _Theor. Comput. Sci._ 296, 3 (2003),

[435–452. https://doi.org/10.1016/S0304-3975(02)00738-7](https://doi.org/10.1016/S0304-3975(02)00738-7)

[58] Harrison Edwards and Amos Storkey. 2015. Censoring representations with an adversary. _arXiv preprint arXiv:1511.05897_ (2015).

[59] Kathleen M Egan, D Trichopoulos, MJ Stampfer, WC Willett, PA Newcomb, A Trentham-Dietz, MP Longnecker, and JA Baron. 1996. Jewish religion

and risk of breast cancer. _The Lancet_ 347, 9016 (1996), 1645–1646.

[60] Danielle Ensign, Sorelle A Friedler, Scott Nevlle, Carlos Scheidegger, and Suresh Venkatasubramanian. 2018. Decision making with limited feedback:

Error bounds for predictive policing and recidivism prediction. In _Proceedings of Algorithmic Learning Theory,_, Vol. 83.

[61] Hasan Erokyar. 2014. Age and gender recognition for speech applications based on support vector machines. (2014).

[62] Simone Fabbrizzi, Symeon Papadopoulos, Eirini Ntoutsi, and Ioannis Kompatsiaris. 2021. A survey on bias in visual datasets. _arXiv preprint_

_arXiv:2107.07919_ (2021).

[63] Eitan Farchi, Ramasuri Narayanam, and Lokesh Nagalapatti. 2021. Ranking Data Slices for ML Model Validation: A Shapley Value Approach. In

_2021 IEEE 37th International Conference on Data Engineering (ICDE)_ . IEEE, 1937–1942.

[64] Sara Feijo. 2018. Here’s what happened when Boston tried to assign students good schools clase to home. (2018).

[65] Siyuan Feng, Olya Kudina, Bence Mark Halpern, and Odette Scharenborg. 2021. Quantifying bias in automatic speech recognition. _arXiv preprint_

_arXiv:2103.15122_ (2021).

[66] Donatella Firmani, Letizia Tanca, and Riccardo Torlone. 2019. Ethical dimensions for data quality. _Journal of Data and Information Quality (JDIQ)_

12, 1 (2019), 1–5.

[67] Sainyam Galhotra, Yuriy Brun, and Alexandra Meliou. 2017. Fairness testing: testing software for discrimination. In _Proceedings of the 2017 11th_

_Joint meeting on foundations of software engineering_ . 498–510.

[68] Tanmay Garg, Sarah Masud, Tharun Suresh, and Tanmoy Chakraborty. 2022. Handling Bias in Toxic Speech Detection: A Survey. _arXiv preprint_

_arXiv:2202.00126_ (2022).

[69] Timnit Gebru, Jamie Morgenstern, Briana Vecchione, Jennifer Wortman Vaughan, Hanna Wallach, Hal Daumé III, and Kate Crawford. 2018.

Datasheets for datasets. _arXiv preprint arXiv:1803.09010_ (2018).

[70] Markos Georgopoulos, James Oldfield, Mihalis A Nicolaou, Yannis Panagakis, and Maja Pantic. 2021. Mitigating demographic bias in facial datasets

with style-based multi-attribute transfer. _International Journal of Computer Vision_ 129, 7 (2021), 2288–2307.

[71] Stephen Gillen, Christopher Jung, Michael Kearns, and Aaron Roth. 2018. Online learning with an unknown fairness metric. _Advances in neural_

_information processing systems_ 31 (2018).

[72] Bruce Glymour and Jonathan Herington. 2019. Measuring the biases that matter: The ethical and casual foundations for measures of fairness in

algorithms. In _Proceedings of the conference on fairness, accountability, and transparency_ . 269–278.

[73] Karan Goel, Albert Gu, Yixuan Li, and Christopher Ré. 2020. Model patching: Closing the subgroup performance gap with data augmentation.

_arXiv preprint arXiv:2008.06775_ (2020).

[74] Gabriel Goh, Andrew Cotter, Maya Gupta, and Michael P Friedlander. 2016. Satisfying real-world goals with dataset constraints. _Advances in_

_Neural Information Processing Systems_ 29 (2016).

[75] Aditya Grover and Jure Leskovec. 2016. node2vec: Scalable feature learning for networks. In _Proceedings of the 22nd ACM SIGKDD international_

_conference on Knowledge discovery and data mining_ . 855–864.

[76] Martyn Hammersley and Roger Gomm. 1997. Bias in social research. _Sociological research online_ 2, 1 (1997), 7–19.

[77] Hui Han, Wenyuan Wang, and Binghuan Mao. 2005. Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning. In

_Advances in Intelligent Computing, International Conference on Intelligent Computing, ICIC 2005, Hefei, China, August 23-26, 2005, Proceedings, Part_


43


Woodstock ’18, June 03–05, 2018, Woodstock, NY Nima Shahbazi, Yin Lin, Abolfazl Asudeh, and H. V. Jagadish


_I (Lecture Notes in Computer Science, Vol. 3644)_, De-Shuang Huang, Xiao-Ping (Steven) Zhang, and Guang-Bin Huang (Eds.). Springer, 878–887.

[https://doi.org/10.1007/11538059_91](https://doi.org/10.1007/11538059_91)

[78] Emma J Harding, Elizabeth S Paul, and Michael Mendl. 2004. Cognitive bias and affective state. _Nature_ 427, 6972 (2004), 312–312.

[79] Moritz Hardt, Eric Price, and Nati Srebro. 2016. Equality of opportunity in supervised learning. In _Advances in neural information processing systems_ .

3315–3323.

[80] Moritz Hardt, Eric Price, and Nathan Srebro. 2016. Equality of Opportunity in Supervised Learning. _ArXiv_ abs/1610.02413 (2016).

[81] Martie G Haselton, Daniel Nettle, and Damian R Murray. 2015. The evolution of cognitive bias. _The handbook of evolutionary psychology_ (2015),

1–20.

[82] David Haussler and Emo Welzl. 1986. Epsilon-nets and simplex range queries. In _Proceedings of the second annual symposium on Computational_

_geometry_ . 61–71.

[83] Xiao Hu, Haobo Wang, Anirudh Vegesana, Somesh Dube, Kaiwen Yu, Gore Kao, Shuo-Han Chen, Yung-Hsiang Lu, George K Thiruvathukal, and

Ming Yin. 2020. Crowdsourcing Detection of Sampling Biases in Image Datasets. In _Proceedings of The Web Conference 2020_ . 2955–2961.

[84] Vasileios Iosifidis and Eirini Ntoutsi. 2018. Dealing with bias via data augmentation in supervised learning scenarios. _Jo Bates Paul D. Clough_

_Robert Jäschke_ 24 (2018).

[85] Hosagrahar V Jagadish, Johannes Gehrke, Alexandros Labrinidis, Yannis Papakonstantinou, Jignesh M Patel, Raghu Ramakrishnan, and Cyrus

Shahabi. 2014. Big data and its technical challenges. _Commun. ACM_ 57, 7 (2014), 86–94.

[86] Nikita Jaipuria, Xianling Zhang, Rohan Bhasin, Mayar Arafa, Punarjay Chakravarty, Shubham Shrivastava, Sagar Manglani, and Vidya N Murali.

2020. Deflating dataset bias using synthetic data augmentation. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_

_Workshops_ . 772–773.

[87] Heinrich Jiang and Ofir Nachum. 2020. Identifying and correcting label bias in machine learning. In _International Conference on Artificial Intelligence_

_and Statistics_ . PMLR, 702–712.

[88] Zhongjun Jin, Mengjing Xu, Chenkai Sun, Abolfazl Asudeh, and HV Jagadish. 2020. MithraCoverage: A System for Investigating Population Bias

for Intersectional Fairness. In _Proceedings of the 2020 ACM SIGMOD International Conference on Management of Data_ . 2721–2724.

[89] Matthew Joseph, Michael Kearns, Jamie Morgenstern, Seth Neel, and Aaron Roth. 2018. Meritocratic fairness for infinite and contextual bandits. In

_Proceedings of the 2018 AAAI/ACM Conference on AI, Ethics, and Society_ . 158–163.

[90] Matthew Joseph, Michael Kearns, Jamie H Morgenstern, and Aaron Roth. 2016. Fairness in learning: Classic and contextual bandits. _Advances in_

_neural information processing systems_ 29 (2016).

[91] Faisal Kamiran and Toon Calders. 2009. Classifying without discriminating. _2009 2nd International Conference on Computer, Control and_

_Communication_ (2009), 1–6.

[92] Faisal Kamiran and Toon Calders. 2011. Data preprocessing techniques for classification without discrimination. _Knowledge and Information_

_Systems_ 33 (2011), 1–33.

[93] Faisal Kamiran, Asim Karim, and Xiangliang Zhang. 2012. Decision Theory for Discrimination-Aware Classification. _2012 IEEE 12th International_

_Conference on Data Mining_ (2012), 924–929.

[94] Toshihiro Kamishima, Shotaro Akaho, and Jun Sakuma. 2011. Fairness-aware Learning through Regularization Approach. _2011 IEEE 11th_

_International Conference on Data Mining Workshops_ (2011), 643–650.

[95] Bo Kang, Jefrey Lijffijt, and Tijl De Bie. 2018. Conditional network embeddings. _arXiv preprint arXiv:1805.07544_ (2018).

[96] Jian Kang and Hanghang Tong. 2021. Fair graph mining. In _Proceedings of the 30th ACM International Conference on Information & Knowledge_

_Management_ . 4849–4852.

[97] Kimmo Karkkainen and Jungseock Joo. 2021. Fairface: Face attribute dataset for balanced race, gender, and age for bias measurement and mitigation.

In _Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision_ . 1548–1558.

[98] Tero Karras, Samuli Laine, and Timo Aila. 2019. A style-based generator architecture for generative adversarial networks. In _Proceedings of the_

_IEEE/CVF Conference on Computer Vision and Pattern Recognition_ . 4401–4410.

[99] Michael Kearns, Seth Neel, Aaron Roth, and Zhiwei Steven Wu. 2019. An empirical study of rich subgroup fairness for machine learning. In _FAT*_ .

ACM.

[100] Ahmad Khajehnejad, Moein Khajehnejad, Mahmoudreza Babaei, Krishna P Gummadi, Adrian Weller, and Baharan Mirzasoleiman. 2022. CrossWalk:

fairness-enhanced node representation learning. In _Proceedings of the AAAI Conference on Artificial Intelligence_, Vol. 36. 11963–11970.

[101] Aditya Khosla, Tinghui Zhou, Tomasz Malisiewicz, Alexei A Efros, and Antonio Torralba. 2012. Undoing the damage of dataset bias. In _European_

_Conference on Computer Vision_ . Springer, 158–171.

[102] Jon Kleinberg, Sendhil Mullainathan, and Manish Raghavan. 2016. Inherent trade-offs in the fair determination of risk scores. _arXiv preprint_

_arXiv:1609.05807_ (2016).

[103] Allison Koenecke, Andrew Nam, Emily Lake, Joe Nudell, Minnie Quartey, Zion Mengesha, Connor Toups, John R Rickford, Dan Jurafsky, and

Sharad Goel. 2020. Racial disparities in automated speech recognition. _Proceedings of the National Academy of Sciences_ 117, 14 (2020), 7684–7689.

[104] Emmanouil Krasanakis, Eleftherios Spyromitros-Xioufis, Symeon Papadopoulos, and Yiannis Kompatsiaris. 2018. Adaptive sensitive reweighting

to mitigate bias in fairness-aware classification. In _Proceedings of the 2018 world wide web conference_ . 853–862.

[105] Matt J Kusner, Joshua Loftus, Chris Russell, and Ricardo Silva. 2017. Counterfactual fairness. _Advances in neural information processing systems_ 30

(2017).


44


Representation Bias in Data: A Survey on Identification and Resolution Techniques Woodstock ’18, June 03–05, 2018, Woodstock, NY


[106] Charlotte Laclau, Ievgen Redko, Manvi Choudhary, and Christine Largeron. 2021. All of the fairness for edge prediction with optimal transport. In

_International Conference on Artificial Intelligence and Statistics_ . PMLR, 1774–1782.

[107] Preethi Lahoti, Krishna P. Gummadi, and Gerhard Weikum. 2019. Operationalizing Individual Fairness with Pairwise Fair Representations. _Proc._

_VLDB Endow._ 13 (2019), 506–518.

[108] Preethi Lahoti, Gerhard Weikum, and Krishna P. Gummadi. 2019. iFair: Learning Individually Fair Data Representations for Algorithmic Decision

Making. _2019 IEEE 35th International Conference on Data Engineering (ICDE)_ (2019), 1334–1345.

[109] [Jennifer Langston. 2015. Who’s a CEO? Google image results can shift gender biases. https://www.washington.edu/news/2015/04/09/whos-a-ceo-](https://www.washington.edu/news/2015/04/09/whos-a-ceo-google-image-results-can-shift-gender-biases/)

[google-image-results-can-shift-gender-biases/.](https://www.washington.edu/news/2015/04/09/whos-a-ceo-google-image-results-can-shift-gender-biases/)

[110] Alyssa Whitlock Lees and Ananth Balashankar. 2019. Fairness Sample Complexity and the Case for Human Intervention. (2019).

[111] Joseph Lemley, Filip Jagodzinski, and Razvan Andonie. 2017. Big Holes in Big Data: A Monte Carlo Algorithm for Detecting Large Hyper-rectangles

in High Dimensional Data. _CoRR_ [abs/1704.00683 (2017). arXiv:1704.00683 http://arxiv.org/abs/1704.00683](http://arxiv.org/abs/1704.00683)

[112] Yi Li and Nuno Vasconcelos. 2019. Repair: Removing representation bias by dataset resampling. In _Proceedings of the IEEE/CVF Conference on_

_Computer Vision and Pattern Recognition_ . 9572–9581.

[[113] M. Lichman. 2013. Adult Income Dataset, UCI Machine Learning Repository. https://archive.ics.uci.edu/ml/datasets/adult.](https://archive.ics.uci.edu/ml/datasets/adult)

[114] Yin Lin, Yifan Guan, Abolfazl Asudeh, and HV Jagadish. 2020. Identifying insufficient data coverage in databases with multiple relations. _Proceedings_

_of the VLDB Endowment_ 13, 12 (2020), 2229–2242.

[115] Bing Liu, Liang-Ping Ku, and Wynne Hsu. 1997. Discovering Interesting Holes in Data. In _Proceedings of the Fifteenth International Joint Conference_

_on Artifical Intelligence - Volume 2_ (Nagoya, Japan) _(IJCAI)_ . Morgan Kaufmann Publishers Inc., 930–935.

[116] Bing Liu, Ke Wang, Lai-Fun Mun, and Xin-Zhi Qi. 1998. Using Decision Tree Induction for Discovering Holes in Data. In _PRICAI (Lecture Notes in_

_Computer Science, Vol. 1531)_, Hing-Yan Lee and Hiroshi Motoda (Eds.). Springer, 182–193.

[117] Chunxi Liu, Michael Picheny, Leda Sarı, Pooja Chitkara, Alex Xiao, Xiaohui Zhang, Mark Chou, Andres Alvarado, Caner Hazirbas, and Yatharth Saraf.

2022. Towards Measuring Fairness in Speech Recognition: Casual Conversations Dataset Transcriptions. In _ICASSP 2022-2022 IEEE International_

_Conference on Acoustics, Speech and Signal Processing (ICASSP)_ . IEEE, 6162–6166.

[118] Yang Liu, Goran Radanovic, Christos Dimitrakakis, Debmalya Mandal, and David C Parkes. 2017. Calibrated fairness in bandits. _arXiv preprint_

_arXiv:1707.01875_ (2017).

[119] Li Lucy and David Bamman. 2021. Gender and representation bias in GPT-3 generated stories. In _Proceedings of the Third Workshop on Narrative_

_Understanding_ . 48–55.

[120] Thomas Manzini, Yao Chong Lim, Yulia Tsvetkov, and Alan W Black. 2019. Black is to criminal as caucasian is to police: Detecting and removing

multiclass bias in word embeddings. _arXiv preprint arXiv:1904.04047_ (2019).

[121] Ninareh Mehrabi, Fred Morstatter, Nripsuta Saxena, Kristina Lerman, and Aram Galstyan. 2021. A survey on bias and fairness in machine learning.

_ACM Computing Surveys (CSUR)_ 54, 6 (2021), 1–35.

[122] Michele Merler, Nalini Ratha, Rogerio S Feris, and John R Smith. 2019. Diversity in faces. _arXiv preprint arXiv:1901.10436_ (2019).

[123] Y. Moskovitch and H. Jagadish. 2021. Patterns Count-Based Labels for Datasets. _2021 IEEE 37th International Conference on Data Engineering (ICDE)_

(2021), 1961–1966.

[124] Yuval Moskovitch and H. V. Jagadish. 2020. COUNTATA: Dataset Labeling Using Pattern Counts. _PVLDB_ 13, 12 (2020), 2829–2832.

[125] Yuval Moskovitch, Jinyang Li, and HV Jagadish. 2022. Bias analysis and mitigation in data-driven tools using provenance. In _Proceedings of the 14th_

_International Workshop on the Theory and Practice of Provenance_ . 1–4.

[126] Fatemeh Nargesian, Abolfazl Asudeh, and HV Jagadish. 2021. Tailoring data source distributions for fairness-aware data integration. _Proceedings of_

_the VLDB Endowment_ 14, 11 (2021), 2519–2532.

[127] Fatemeh Nargesian, Abolfazl Asudeh, and H. V. Jagadish. 2022. Responsible Data Integration: Next-generation Challenges. _SIGMOD_ (2022).

[128] Mark EJ Newman. 2003. Mixing patterns in networks. _Physical review E_ 67, 2 (2003), 026126.

[129] Jerzy Neyman and Egon Sharpe Pearson. 1936. Contributions to the theory of testing statistical hypotheses. _Statistical Research Memoirs_ (1936).

[130] Laura Niss, Yuekai Sun, and Ambuj Tewari. 2022. Achieving Representative Data via Convex Hull Feasibility Sampling Algorithms. _arXiv preprint_

_arXiv:2204.06664_ (2022).

[131] Eirini Ntoutsi, Pavlos Fafalios, Ujwal Gadiraju, Vasileios Iosifidis, Wolfgang Nejdl, Maria-Esther Vidal, Salvatore Ruggieri, Franco Turini, Symeon

Papadopoulos, Emmanouil Krasanakis, et al . 2020. Bias in data-driven artificial intelligence systems—An introductory survey. _Wiley Interdisciplinary_

_Reviews: Data Mining and Knowledge Discovery_ 10, 3 (2020), e1356.

[132] Alexandra Olteanu, Carlos Castillo, Fernando Diaz, and Emre Kiciman. 2019. Social data: Biases, methodological pitfalls, and ethical boundaries.

_Frontiers in Big Data_ 2 (2019), 13.

[133] Kalia Orphanou, Jahna Otterbacher, Styliani Kleanthous, Khuyagbaatar Batsuren, Fausto Giunchiglia, Veronika Bogina, Avital Shulner Tal, Alan

Hartman, and Tsvi Kuflik. 2021. Mitigating Bias in Algorithmic Systems-A Fish-Eye View. _ACM Computing Surveys (CSUR)_ (2021).

[134] Orestis Papakyriakopoulos, Simon Hegelich, Juan Carlos Medina Serrano, and Fabienne Marco. 2020. Bias in word embeddings. In _Proceedings of_

_the 2020 conference on fairness, accountability, and transparency_ . 446–457.

[135] Ji Ho Park, Jamin Shin, and Pascale Fung. 2018. Reducing gender bias in abusive language detection. _arXiv preprint arXiv:1808.07231_ (2018).

[136] Eliana Pastor, Luca de Alfaro, and Elena Baralis. 2021. Looking for Trouble: Analyzing Classifier Behavior via Pattern Divergence. In _Proceedings of_

_the 2021 International Conference on Management of Data_ . 1400–1412.


45


Woodstock ’18, June 03–05, 2018, Woodstock, NY Nima Shahbazi, Yin Lin, Abolfazl Asudeh, and H. V. Jagadish


[137] Dana Pessach and Erez Shmueli. 2022. A Review on Fairness in Machine Learning. _ACM Computing Surveys (CSUR)_ 55, 3 (2022), 1–44.

[138] Geoff Pleiss, M. Raghavan, Felix Wu, Jon M. Kleinberg, and Kilian Q. Weinberger. 2017. On Fairness and Calibration. _ArXiv_ abs/1709.02012 (2017).

[139] Romila Pradhan, Jiongli Zhu, Boris Glavic, and Babak Salimi. 2021. Interpretable data-based explanations for fairness debugging. _arXiv preprint_

_arXiv:2112.09745_ (2021).

[140] Tahleen Rahman, Bartlomiej Surma, Michael Backes, and Yang Zhang. 2019. Fairwalk: Towards fair graph embedding. (2019).

[141] Kumar Rakesh, Subhangi Dutta, and Kumara Shama. 2011. Gender Recognition using speech processing techniques in LABVIEW. _International_

_Journal of Advances in Engineering & Technology_ 1, 2 (2011), 51.

[142] Svetlana Sagadeeva and Matthias Boehm. 2021. SliceLine: Fast, Linear-Algebra-based Slice Finding for ML Model Debugging. In _Proceedings of the_

_2021 International Conference on Management of Data_ . 2290–2299.

[143] Babak Salimi, Luke Rodriguez, Bill Howe, and Dan Suciu. 2019. Interventional fairness: Causal database repair for algorithmic fairness. In

_Proceedings of the 2019 International Conference on Management of Data_ . 793–810.

[144] Beatrice Savoldi, Marco Gaido, Luisa Bentivogli, Matteo Negri, and Marco Turchi. 2021. Gender bias in machine translation. _Transactions of the_

_Association for Computational Linguistics_ 9 (2021), 845–874.

[145] Nina Schaaf, Omar de Mitri, Hang Beom Kim, Alexander Windberger, and Marco F Huber. 2021. Towards Measuring Bias in Image Classification.

In _International Conference on Artificial Neural Networks_ . Springer, 433–445.

[146] Ben Schmidt. 2015. Rejecting the gender binary: a vector-space operation. _Ben’s Bookworm Blog_ (2015).

[147] LS Shapley. 1952. PROJECT RAND. (1952).

[148] Shubham Sharma, Yunfeng Zhang, Jesús M Ríos Aliaga, Djallel Bouneffouf, Vinod Muthusamy, and Kush R Varshney. 2020. Data augmentation for

discrimination prevention and bias disambiguation. In _Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society_ . 358–364.

[149] Shubhanshu Shekhar, Greg Fields, Mohammad Ghavamzadeh, and Tara Javidi. 2021. Adaptive sampling for minimax fair classification. _Advances_

_in Neural Information Processing Systems_ 34 (2021), 24535–24544.

[150] Suraj Shetiya, Ian P. Swift, Abolfazl Asudeh, and Gautam Das. 2022. Fairness-Aware Range Queries for Selecting Unbiased Data. In _2022 IEEE 38th_

_International Conference on Data Engineering (ICDE)_ . IEEE.

[151] Indro Spinelli, Simone Scardapane, Amir Hussain, and Aurelio Uncini. 2021. Fairdrop: Biased edge dropout for enhancing fairness in graph

representation learning. _IEEE Transactions on Artificial Intelligence_ 3, 3 (2021), 344–354.

[152] Julia Stoyanovich, Bill Howe, and HV Jagadish. 2020. Responsible data management. _Proceedings of the VLDB Endowment_ 13, 12 (2020).

[153] Seymour Sudman. 1976. _Applied sampling_ . Technical Report. Academic Press New York.

[154] Chenkai Sun, Abolfazl Asudeh, HV Jagadish, Bill Howe, and Julia Stoyanovich. 2019. Mithralabel: Flexible dataset nutritional labels for responsible

data science. In _Proceedings of the 28th ACM International Conference on Information and Knowledge Management_ . 2893–2896.

[155] Tony Sun, Andrew Gaut, Shirlyn Tang, Yuxin Huang, Mai ElSherief, Jieyu Zhao, Diba Mirza, Elizabeth Belding, Kai-Wei Chang, and William Yang

Wang. 2019. Mitigating gender bias in natural language processing: Literature review. _arXiv preprint arXiv:1906.08976_ (2019).

[156] Harini Suresh and John Guttag. 2021. A framework for understanding sources of harm throughout the machine learning life cycle. In _Equity and_

_Access in Algorithms, Mechanisms, and Optimization_ . 1–9.

[157] Ki Hyun Tae and Steven Euijong Whang. 2021. Slice tuner: A selective data acquisition framework for accurate and fair machine learning models.

In _Proceedings of the 2021 International Conference on Management of Data_ . 1771–1783.

[158] Antonio Torralba and Alexei A Efros. 2011. Unbiased look at dataset bias. In _CVPR 2011_ . IEEE, 1521–1528.

[159] Laurens Van der Maaten and Geoffrey Hinton. 2008. Visualizing data using t-SNE. _Journal of machine learning research_ 9, 11 (2008).

[160] Eva Vanmassenhove, Christian Hardmeier, and Andy Way. 2019. Getting gender right in neural machine translation. _arXiv preprint arXiv:1909.05088_

(2019).

[161] Pranav Narayanan Venkit and Shomir Wilson. 2021. Identification of bias against people with disabilities in sentiment analysis and toxicity

detection models. _arXiv preprint arXiv:2111.13259_ (2021).

[162] Sahil Verma and Julia Sass Rubin. 2018. Fairness Definitions Explained. _2018 IEEE/ACM International Workshop on Software Fairness (FairWare)_

(2018), 1–7.

[163] Christina Wadsworth, Francesca Vera, and Chris Piech. 2018. Achieving fairness through adversarial learning: an application to recidivism

prediction. _arXiv preprint arXiv:1807.00199_ (2018).

[164] Angelina Wang, Arvind Narayanan, and Olga Russakovsky. 2020. REVISE: A tool for measuring and mitigating bias in visual datasets. In _European_

_Conference on Computer Vision_ . Springer, 733–751.

[165] Blake Woodworth, Suriya Gunasekar, Mesrob I Ohannessian, and Nathan Srebro. 2017. Learning non-discriminatory predictors. In _Conference on_

_Learning Theory_ . PMLR, 1920–1953.

[166] Depeng Xu, Yongkai Wu, Shuhan Yuan, Lu Zhang, and Xintao Wu. 2019. Achieving causal fairness through generative adversarial networks. In

_Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence_ .

[167] Depeng Xu, Shuhan Yuan, Lu Zhang, and Xintao Wu. 2018. FairGAN: Fairness-aware Generative Adversarial Networks. _2018 IEEE International_

_Conference on Big Data (Big Data)_ (2018), 570–575.

[168] Seyma Yucer, Samet Akçay, Noura Al-Moubayed, and Toby P Breckon. 2020. Exploring racial bias within face recognition via per-subject

adversarially-enabled data augmentation. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops_ . 18–19.


46


Representation Bias in Data: A Survey on Identification and Resolution Techniques Woodstock ’18, June 03–05, 2018, Woodstock, NY


[169] Muhammad Bilal Zafar, Isabel Valera, Manuel Gomez-Rodriguez, and Krishna P. Gummadi. 2017. Fairness Constraints: Mechanisms for Fair

Classification. In _AISTATS_ .

[170] Richard S. Zemel, Ledell Yu Wu, Kevin Swersky, Toniann Pitassi, and Cynthia Dwork. 2013. Learning Fair Representations. In _ICML_ .

[171] Brian Hu Zhang, Blake Lemoine, and Margaret Mitchell. 2018. Mitigating unwanted biases with adversarial learning. In _Proceedings of the 2018_

_AAAI/ACM Conference on AI, Ethics, and Society_ . 335–340.

[172] Hantian Zhang, Xu Chu, Abolfazl Asudeh, and Shamkant B Navathe. 2021. Omnifair: A declarative system for model-agnostic group fairness in

machine learning. In _Proceedings of the 2021 International Conference on Management of Data_ . 2076–2088.

[173] Jieyu Zhao, Tianlu Wang, Mark Yatskar, Vicente Ordonez, and Kai-Wei Chang. 2018. Gender bias in coreference resolution: Evaluation and

debiasing methods. _arXiv preprint arXiv:1804.06876_ (2018).

[174] Jieyu Zhao, Yichao Zhou, Zeyu Li, Wei Wang, and Kai-Wei Chang. 2018. Learning gender-neutral word embeddings. _arXiv preprint arXiv:1809.01496_

(2018).


47


