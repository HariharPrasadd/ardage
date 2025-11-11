## **Predicting drug response of tumors from integrated** **genomic profiles by deep neural networks**

**Yu-Chiao Chiu** **[1]** **, Hung-I Harry Chen** **[1,2]** **, Tinghe Zhang** **[2]** **, Songyao Zhang** **[2,3]** **, Aparna**
**Gorthi** **[1]** **, Li-Ju Wang** **[1]** **, Yufei Huang** **[2,4§]** **, Yidong Chen** **[1,4§]**


1 Greehey Children’s Cancer Research Institute, University of Texas Health Science Center


at San Antonio, San Antonio, TX 78229, USA


2 Department of Electrical and Computer Engineering, University of Texas at San Antonio,


San Antonio, TX 78249, USA


3 Laboratory of Information Fusion Technology of Ministry of Education, School of


Automation, Northwestern Polytechnical University, Xi’an, Shaanxi 710072, China


4 Department of Epidemiology and Biostatistics, University of Texas Health Science Center


at San Antonio, San Antonio, TX 78229, USA


§ Corresponding authors


Email addresses:


YH: Yufei.Huang@utsa.edu


YC: ChenY8@uthscsa.edu


                           - 1 

## **Abstract**

**Background**


The study of high-throughput genomic profiles from a pharmacogenomics viewpoint has


provided unprecedented insights into the oncogenic features modulating drug response. A


recent study screened for the response of a thousand human cancer cell lines to a wide


collection of anti-cancer drugs and illuminated the link between cellular genotypes and


vulnerability. However, due to essential differences between cell lines and tumors, to date


the translation into predicting drug response in tumors remains challenging. Recently,


advances in deep neural networks (DNNs) have revolutionized bioinformatics and


introduced new techniques to the integration of genomic data. Its application on


pharmacogenomics may fill the gap between genomics and drug response and improve the


prediction of drug response in tumors.


**Results**


We proposed a DNN model to predict drug response based on mutation and expression


profiles of a cancer cell or a tumor. The model contains three subnetworks, i) a mutation


encoder pre-trained using a large pan-cancer dataset to abstract core representations of


high-dimension mutation data, ii) a pre-trained expression encoder, and iii) a drug response


predictor network integrating the first two subnetworks. Given a pair of mutation and


expression profiles, the model predicts IC 50 values of 265 drugs. We trained and tested the


model on a dataset of 622 cancer cell lines and achieved an overall prediction performance


of mean squared error at 1.96 (log-scale IC 50 values). The performance was superior in


prediction error or stability than two classical methods (linear regression and support vector


                           - 2 

machine) and four analog DNNs of our model, including DNNs built without TCGA pre

training, partly replaced by principal components, and built on individual types of input


data. We then applied the model to predict drug response of 9,059 tumors of 33 cancer


types. Using per-cancer and pan-cancer settings, the model predicted both known,


including EGFR inhibitors in non-small cell lung cancer and tamoxifen in ER+ breast


cancer, and novel drug targets, such as vinorelbine for _TTN_ -mutated tumors. The


comprehensive analysis further revealed the molecular mechanisms underlying the


resistance to a chemotherapeutic drug docetaxel in a pan-cancer setting and the anti-cancer


potential of a novel agent, CX-5461, in treating gliomas and hematopoietic malignancies.


**Conclusions**


Here we present, as far as we know, the first DNN model to translate pharmacogenomics


features identified from _in vitro_ drug screening to predict the response of tumors. The


results covered both well-studied and novel mechanisms of drug resistance and drug targets.


Our model and findings improve the prediction of drug response and the identification of


novel therapeutic options.


**Keywords:** deep neural networks, pharmacogenomics, drug response prediction, Cancer


Cell Line Encyclopedia, Genomics of Drug Sensitivity in Cancer, The Cancer Genome


Atlas


                           - 3 

## **Background**

Due to tumor heterogeneity and intra-tumor sub-clones, an accurate prediction of drug


response and an identification of novel anti-cancer drugs remain challenging tasks [1, 2].


Pharmacogenomics, an emerging field studying how genomic alterations and


transcriptomic programming determine drug response, represents a potential solution [3,


4]. For instance, recent reports identified mutation profiles associated with drug response


both in tumor type-specific and pan-cancer manners [5, 6]. As drug response data of large


patient cohorts are scarcely available, large-scale cell line-based screening can greatly


facilitate the study of pharmacogenomics in cancer. Recently, the Genomics of Drug


Sensitivity in Cancer (GDSC) Project proposed a comprehensively landscape of drug


response of ~1,000 human cancer cell lines to 265 anti-cancer drugs and unveiled crucial


oncogenic aberrations related to drug sensitivity [7, 8]. Because of the fundamental


differences between _in vitro_ and _in vivo_ biological systems, a translation of


pharmacogenomics features derived from cells to the prediction of drug response of tumors


is to our knowledge not yet realized.


Deep learning (DL) is the state-of-the-art machine learning technology for learning


knowledge from complex data and making accurate predictions. It features the ability to


learn the representation of data without the need for prior knowledge and an assumption


on data distributions. The DL technology has been successfully applied to bioinformatics


studies of regulatory genomics, such as predicting binding motifs [9], investigating DNA


variants [10], deciphering single-cell omics [11, 12], and extraction of genomics features


for survival prediction [13]. In pharmaceutical and pharmacogenomics research, reports


have shown its ability to predict drug-target interactions [14], screen for novel anti-cancer


                           - 4 

drugs [15], and predict drug synergy [16]. Nevertheless, data complexity and the


requirement of large training datasets have limited its application to integrate genomics


data and comprehensively predict drug response, hindering the translation to precision


oncology.


Addressing the unmet demands, the present study is aimed to predict the response of


tumors to anti-cancer drugs based on genomic profiles. We designed a deep neural network


(DNN) model to learn the genetic background from high-dimensional mutation and


expression profiles using the huge collection of tumors of The Cancer Genome Atlas


(TCGA). The model was further trained by the pharmacogenomics data developed in


human cancer cell lines by the GDSC Project and their corresponding genomic and


transcriptomic alteration, and finally applied to TCGA data again to predict drug response


of tumors. Collectively, this study demonstrated a novel DL model that bridges cell line

based pharmacogenomics knowledge via tumor genomic and transcriptomic abstraction to


predict tumors’ response to compound treatment.


                           - 5 

## **Methods**

**Datasets**


We downloaded gene-level expression data of 935 cell lines of the Cancer Cell Line


Encyclopedia (CCLE) and 11,078 TCGA pan-cancer tumors from the CTD [2] Data Portal


[17] and UCSC TumorMap [18], respectively. Given the total numbers of cell lines, tumors,


and genes as _C_, _T_, _G_, respectively, we metricized the expression data by ! [""#!] =



,,.//01 + 1, where )*+,,.



%&' ( )*+,,.



//01 is the number of transcripts per million of gene _g_



( ' ∈1, 5 ) in cell line _c_ ( 6 ∈1, 7 ), and ! [8"9:] = %&' ( )*+,,;</=> + 1, where



)*+,,;</=> denotes the number of transcripts per million of the same gene in tumor _t_ () ∈


1, ? ). Genes with low information burden (mean < 1 or st. dev. < 0.5) among TCGA


samples were removed. Mutation Annotation Format (MAF) files of mutation data were


downloaded directly from CCLE (1,463 cells) [19, 20] and TCGA databases (10,166


tumors). Here we only considered four types of nonsynonymous mutations, including


missense and nonsense mutations, frameshift insertions and deletions. Thus, we had binary



matrices of @ [""#!] = +
,,.



//01 and @ 8"9: = +

,,.,,;



</=>

,,;, where +,,.



//01
and +

,,.,,;



</=> are the



mutation states (1 for mutation and 0 for wildtype) of gene _g_ in _c_ and _t_, respectively. Genes


with no mutations in CCLE and TCGA samples were eliminated.


We also downloaded drug response data of 990 CCLE cell lines to 265 anti-cancer


drugs measured by the half maximal inhibitory concentration (IC 50 ) from the GDSC Project


[7]. IC 50 were measured in µM and represented in log scale ( _i.e._, A" [""#!] = %&' BC D6 E,.//01,


with _d_ denoting the _d_ -th drug and F ∈1, G ) and missing data were imputed by a weighted


mean of IC 50 of 5 nearest drugs using R packages VIM and laeken [21, 22]. In this study,


                           - 6 

we analyzed 622 cell lines with available expression, mutation, and IC 50 data and 9,059


tumors with expression and mutation profiles.


**General settings of DNNs and computation environment**


DNN training in this study were performed using the python library Keras 1.2.2 with


TensorFlow backend. We used fully (or densely) connected layers for all networks. At a


neuron _j_, its output H I is calculated by


H I = J L K LI M L + N I (1)


, where M L is the output of neuron _i_ at the previous layer of _j_, K LI and N I denote the synaptic


weight and bias, respectively, and J represents an activation function. The notation of all


neurons at a layer can thus be written as


O = J PQ + R . (2)


During training, synaptic weights and biases are adjusted to minimize a loss function. We


hereafter refer to the two parameters as synaptic parameters because they represent the


model and can be used to transfer a learned model to another. In this study, models were


optimized using the Adam optimizer with a loss function of mean squared error (MSE).


We used the He’s uniform distribution [23] to initialize autoencoders and the Prediction (P)


network, while the mutation encoder (M enc ) and expression encoder (E enc ) in the complete


model were initialized by the synaptic parameters learned from the pre-training on TCGA


data. Neuron activation function was set as rectified linear unit (ReLU) except for the


output layer of P as linear in order to better fit the distribution of log-scale IC 50 .


                           - 7 

**Overview of the proposed DNN model**


The proposed DNN model was developed to predict IC 50 values based on genomic profiles


of a cell or a tumor. Given the pair of mutation and expression vectors of sample


_c_, @ [""#!] :, 6, ! [""#!] :, 6, the model predicts a _D_ -length vector of IC 50, A" [""#!] 6, as


an output. As shown in Figure 1, the model is composed of three networks: i) a mutation


encoder (M enc ), ii) an expression encoder (E enc ), and iii) a prediction feedforward network


(P). The first and second components are the encoding parts of two autoencoders pre

trained using TCGA data to learn the high-order features of mutation and expression data


into a lower dimensional representation. The encoded representation of mutation and


expression profiles were linked into P and the entire model was trained on CCLE data to


make prediction of IC 50 values. Details of our model are described below.


**Pre-training of mutation and expression encoders**


Autoencoder is an unsupervised DL architecture that includes an asymmetric pair of


encoder and decoder. By minimizing the loss between input and reconstructed ( _i.e._,


decoded) data, it reduces the dimension of complex data and captures crucial features at


the bottleneck layer (the layer between encoder and decoder) (Figure 1B, top and bottom


panels). We pre-trained an autoencoder on each of the TCGA mutation and expression


datasets to optimize the capability to capture high-order features. To determine the


optimized architecture, we adopted a hyper-parameter optimization method, namely


hyperas [24], to select i) number of neurons at the 1 [st] layer (4096, 2048, or 1024), ii)


number of neurons at the 2 [nd] layer (512, 256, or 128), iii) number of neurons at the 3 [rd] layer


(the bottleneck layer; 64, 32, or 16), and iv) batch size (128 or 64). Each combination was


                           - 8 

trained for 20 epochs; the best-performing model was re-run for 100 epochs and the


synaptic parameters were saved.


**Complete prediction network**


In our complete model, encoders of the two optimized autoencoders, _i.e._, M enc and E enc,


were linked to P to make predictions of IC 50 (Figure 1). P is a 5-layer feedforward neural


network, including the first layer merging output neurons of the two encoders, three fully


connected layers, and the last layer of _d_ neurons generating IC 50 values of _d_ drugs (Figure


1B, orange box). In the complete model, architecture (number of layers and neurons at each


layer) of M enc and E enc were fixed; their synaptic parameters were initialized using the


parameters obtained from pre-training in TCGA and updated during the training process.


P was randomly initialized. We trained the entire model using CCLE data, with 80%, 10%,


and 10% of samples as training, validation, and testing sets, respectively. We note the


validation dataset was used to update model parameters but to stop the training process


when the loss in validation set had stopped decreasing for 3 consecutive epochs to avoid


model overfitting. Performance of the model was evaluated using the testing samples, _i.e._,


UVW A" [""#!] :, " XYZX, A" [""#!] :, " XYZX, where " XYZX denotes the test set of cell lines.


We applied the final model to predict drug response of TCGA tumors. For a tumor _t_,


@ [8"9:] :, ), ! [8"9:] :, ) was fed into the model and A" [8"9:] :, ) was calculated. A high


predicted IC 50 indicates an adverse response of a patient to the corresponding drug.


**Comparison to other model designs**


Performance of the complete neural network model was compared to four different DNN


designs. First, to assess the effect of TCGA pre-training on M enc and E enc, we randomly


                           - 9 

initialized both encoders using the He’s uniform distribution and calculated MSE of the


entire model. Second, dimension reduction of the M enc and E enc networks was replaced by


principal component analysis (PCA). Last two models were built without M enc or E enc to


study whether they jointly improved the performance. In each iteration, CCLE samples


were randomly assigned to training (80%), validation (10%), and testing (10%) and each


model was trained and tested. Performance in terms of the number of consumed epochs


and MSE in IC 50 were summarized and compared across the 100 iterations. We also


analyzed two classical prediction methods, multivariate linear regression and regularized


support vector machine (SVM). For each method, top 64 principle components of


mutations and gene expression were merged to predict IC 50 values of all (using linear


regression) or individual drugs (SVM).


                         - 10 

## **Results and Discussion**

**Model construction and evaluation in CCLE**


The study is aimed to predict drug response (measured as log-scale IC 50 values) using


genome-wide mutation and expression profiles. We included mutation and expression


profiles of 622 CCLE cell lines of 25 tissue types and 9,059 TCGA tumors of 33 cancer


types. After data preprocessing, 18,281 and 15,363 genes with mutation and expression


data, respectively, available in both CCLE and TCGA samples were analyzed. Log-scale


IC 50 values of all cell lines in response to 265 anti-cancer drugs were collected from the


GDSC Project [7]. After imputation of missing values, the range of log IC 50 was from -9.8


to 12.8 with a standard deviation of 2.6 (Figure 2A). We designed a DNN model with three


building blocks: 4-layer M enc and 4-layer E enc for capturing high-order features and


reducing dimensions of mutation and expression data, and a 5-layer prediction network P


integrating the mutational and transcriptomic features to predict IC 50 of multiple drugs


(Figure 1). To make the best use of the large collection of TCGA pan-cancer data, we pre

trained an autoencoder for each data type and extracted the encoders, M enc (number of


neurons at each layer, 18,281, 1,024, 256, and 64) and E enc (15,363, 1,024, 256, and 64),


to construct our final model (detailed in Methods). Output neurons of the two encoders


were linked to P (number of neurons at each layer, 64+64, 128, 128, 128, and 265), of


which the last layer outputs predicted IC 50 . Architecture of the complete neural networks


is shown in Figure 1B.


After pre-training M enc and E enc components, we trained the entire model using 80% of


CCLE samples together with a validation set of 10% of samples to avoid overfitting. The


remaining samples (64 cells; 16,960 cell-drug combinations) were used for testing. The


                         - 11 

model achieved an overall MSE in IC 50 of 1.53, corresponding to 1.48 and 1.98 in


training/validation and testing data, respectively. Generally, the distribution of predicted


IC 50 was similar to original data (Figure 2A-B), while the two modes of original data


seemed to be enhanced (highlighted in Figure 2A). In both training/validation and testing


data, the prediction was highly consistent to the true data in terms of IC 50 values (Pearson


correlation; [ \ ) and rank of drugs (Spearman correlation; [ ] ) of a sample ( [ \ ∈


0.70,0.96, [ ] ∈0.62,0.95, and all _P_ -values < 1.0×10 [-29] ; Figure 2C-D). Of note,


correlations achieved in training/validation and testing samples were highly comparable


(Figure 2C-D), confirming the performance of our model.


**Performance comparisons to other designs**


To test the stability of our model, we ran 100 training processes each of which training,


validation, and testing cells were reselected. Overall, the model converged in 14.0 epochs


(st. dev., 3.5; Table 1) and achieved an MSE of 1.96 in testing samples (st. dev., 0.13;


Figure 2E and Table 1). We compared the performance to linear regression, SVM, and four


analog DNNs of our model, including random initialization (identical architecture, but


without TCGA pre-training of M enc and E enc ), PCA (M enc and E enc each replaced by top 64


principal components of mutation and expression data), M enc only (E enc removed from the


model), and E enc only (M enc removed from the model). The two classical methods seemed


to suffer from high MSE in testing samples (10.24 and 8.92 for linear regression and SVM,


respectively; Table 1). Our model also outperformed DNNs with random initialization and


PCA in MSE (difference in medians, 0.34 and 0.48; Figure 2E and Table 1) and stability


(st. dev. of MSE in testing samples = 0.13, 1.21, and 0.17 for our model, random


initialization, and PCA, respectively; Figure 2E). While the E enc -only model achieved


                         - 12 

similar performance to our model (difference in medians = 0.0042; Figure 2E and Table


1), the addition of M enc seemed to bring faster convergence (difference in medians = 3;


Table 1). Our data echoed the biological premise that gene expressions are more directly


linked to biological functions and thus richer in information burden than mutations.


**Associations of gene mutations to predicted drug response in TCGA – per-cancer**


**study**


In search of effective anti-cancer drugs in tumors, we applied the constructed model


directly to predict the response of 9,059 TCGA samples to the 265 anti-cancer drugs. The


predicted IC 50 values followed a similar distribution to CCLE cells (Figure 2A, blue line).


Realizing the different nature of cell lines and tumors, we started by examining several


drugs with well-known target genes. As shown in Figure 3A, breast invasive carcinoma


(BRCA) with positive estrogen receptor (ER; assessed by immunohistochemistry by


TCGA) responded to a selective estrogen receptor modulator, tamoxifen, significantly


better than ER-negative patients ( _t_ -test _P_ = 2.3×10 [-4] ). Also, two EGFR inhibitors, afatinib


and gefitinib, achieved better performance in non-small cell lung cancers (NSCLC) with


mutated _EGFR_ ( _P_ = 2.0×10 [-7 ] and 6.6×10 [-3] ). While the promising results on these well

characterized drugs showed the applicability of our model to tumors, we noted that the


magnitude of differences in predicted IC 50 levels was modest, underlining the fundamental


differences between cell lines and tumors. In order to prioritize mutations underlying drug


response, we systematically analyzed all cancer–mutation–drug combinations and tested


the significance of differences in IC 50 between samples with and without a mutation for


each cancer. Here only genes with a mutation rate higher than 10% and harbored by at least


10 patients in a cancer were analyzed. With a stringent criterion of Bonferroni-adjusted _t_ 

                         - 13 

test _P_ < 1.0×10 [-5], we identified a total of 4,453 significant cancer–mutation–drug


combinations involving 256 drugs and 169 cancer–mutation combinations (Figure 3B).


The top three combinations were _TP53_ mutations in lung adenocarcinoma (LUAD;


modulating response to 235 drugs), lung squamous cell carcinoma (LUSC; 228 drugs), and


stomach adenocarcinoma (STAD; 224 drugs) (Table 2). _TP53_ was one of the most


frequently mutated and well-studied genes in many cancers. The mutation has been shown


to be associated with cancer stem cells and resistance functions and thus regulates drug


resistance [25, 26]. For instance, our data indicated its associations with resistance of a


PI3Kβ inhibitor, TGX221, in 9 cancers including low-grade glioma (LGG; mean difference


in IC 50 (ΔIC 50 ) = 0.95; _P_ = 2.2×10 [-109] ; Figure 3C) and resistance of vinorelbine in BRCA


(ΔIC 50 = 0.68; _P_ = 7.4×10 [-71] ; Figure 3C) and 6 other cancers. We also identified gene


mutations that sensitized tumors to a large number of drugs, such as _IDH1_ (138 drugs;


Table 2). _IDH1_ was the most frequently mutated gene in LGG (77.3% in our data; Table


2) and known to regulate cell cycle of glioma cells and enhance the response to


chemotherapy [27]. Our finding agreed with the report and showed that _IDH1_ mutation


dramatically reduced IC 50 of chemotherapeutic agents, e.g., doxorubicin in LGG (ΔIC 50 =


-0.85; _P_ = 3.6×10 [-71] ; Figure 3C).


**Associations of gene mutations to predicted drug response in TCGA – pan-cancer**


**study**


We also carried out a study to explore how gene mutations affect drug response in a pan

cancer setting. The analysis was focused on 11 genes with mutation rates higher than 10%


across all TCGA samples (Table 3). Using an identical criterion, we identified 2,119


significant mutation–drug pairs composed of 256 drugs, among which 1,882 (88.8%) and


                         - 14 

237 (11.2%) were more resistant and sensitive in mutated samples, respectively (Figure 4A


and Table 3). _TP53_ (251 drugs), _CSMD3_ (223), and _SYNE1_ (218), _TTN_ (206), and _RYR2_


(199) were the top drug response-modulating genes (Table 3). Among them, _TP53_ (9


sensitive and 242 resistant drugs) and _TTN_ mutations (44 and 162) were associated with


the most numbers of resistant and sensitive drugs, respectively (Table 3). Thus, we further


investigated the drug response and their association with status of the 2 genes. Many of the


drugs with large _TP53_ mutations-modulated changes in ΔIC 50 (|ΔIC 50 | ≥ 0.7; Figure 4A-B)


were previously studied in different cancer types by _in vitro_ models. For instance, wildtype


_TP53_ is required in the anti-cancer actions of CX-5461 [28, 29] and sorafenib [30] (both _P_


of ΔIC 50 ~0 in our data; Figure 4B), sensitizes various cancer cells to bortezomib [31] ( _P_


= 4.4×10 [-308] ; Figure 4B), and enhances phenformin-induced growth inhibition and


apoptosis [32] ( _P_ =2.0×10 [-241] ; Figure 4B). As for previously less explored _TTN_ mutations,


the longest gene in human genome known to carry large variations, our data indicated that


perhaps _TNN_ acts as a marker gene of tumors sensitized to chemotherapeutic agents such


as vinorelbine ( _P_ ~0; Figure 4C) and a potential anti-cancer drug epothilone B ( _P_ =2.5×10 [-]


253 ; Figure 4C). Taken together findings from our per- and pan-cancer studies, we have


demonstrated the applicability of our model to predict drug response of tumors and ability


to unveil novel and well-studied genes modulating drug response in cancer.


**Pharmacogenomics analysis of docetaxel and CX-5461 in TCGA**


To unveil the pharmacogenomics landscape of drugs, a comprehensive study of mutation


and expression profiles associated with resistance of a drug in a pan-cancer setting was


carried out. Here we took two drugs as demonstrating examples, a widely used


chemotherapeutic agent docetaxel and a novel anti-cancer drug CX-5461 currently under


                         - 15 

investigation in several cancers. For each drug, pan-cancer patients predicted to be very


sensitive and resistant (with IC 50 in bottom and top 1%, n = 91 in each group; Figure 5A,


left panel) were compared for cancer type composition, mutation rates, and differential


gene expression. Top cancer types of docetaxel-sensitive patients were among esophageal


carcinoma (ESCA; 25.3%), cervical and endocervical cancer (CESC; 13.2%), and head


and neck squamous cell carcinoma (HNSC; 9.9%) (Figure 5B, left panel), while top


resistant patients were mainly liver hepatocellular carcinoma (LIHC; 42.9%), LGG


(26.4%), and glioblastoma multiforme (GBM; 12.1%) (Figure 5B, left panel). Top 10 gene


with most changed mutation rates between the two groups of patients are listed in Figure


5C. On average, each sensitive tumor harbored 2.7 mutations among these genes, much


higher than 0.51 observed in the resistant group (Figure 5C, left panel), implying tumors


with higher mutation burdens in crucial genes may be more vulnerable to the treatment. Of


note, a great majority of the most significantly differentially expressed genes were


upregulated in sensitive patients (Figure 5C, left panel). We performed functional


annotation analysis of the top 300 genes in Gene Ontology terms of biological processes


and molecular functions using the Database for Annotation, Visualization and Integrated


Discovery (DAVID) v6.7 [33, 34]. While we did not observe any cluster of functions


related to microtubule, through which docetaxel physically binds to the cell and regulate


the cell cycle [35], these drug sensitivity-related genes were indeed predominantly enriched


in functions governing the mitotic cell cycle (Table 4). The observation largely reflected


the nature of the chemotherapeutic agent to target highly proliferative cells and the


dependence of drug response on the ability to pass cell-cycle checkpoints. In addition to


docetaxel, we analyzed a novel anti-cancer agent, CX-5461. This inhibitor of ribosomal


                         - 16 

RNA synthesis has been shown with anti-cancer properties in cancer cells [36, 37] and is


now under phase I/II clinical trial in solid tumors (NCT number, NCT02719977). In


hematopoietic malignancies, it was recently shown to outperform standard chemotherapy


regimen in treating aggressive acute myeloid leukemia (LAML) [29], and its anti-cancer


effects were dependent on wild-type _TP53_ [28, 29]. Concordantly, in our data, LAML and


lymphoid neoplasm diffuse large B-cell lymphoma (DLBC) jointly accounted for 45.1%


(41.8% and 3.3%) of patients predicted be respond extremely well to CX-5461 (Figure 5A

B, right panels). Of note, LGG comprised another 48.4% of the sensitive tumors (Figure


5B, right panel). Nine of the top 10 differentially mutated genes were enriched in the


resistant group and leaded by _TP53_ mutations (mutation rate, 95.6% in resistant vs. 13.2%


in sensitive patients; Figure 5C, right panel), echoing data from our pan-cancer analysis


(Figure 4A-B) and previous _in vitro_ and _in vivo_ investigations [28, 29]. _IDH1_ was the only


gene preferentially mutated in sensitive tumors and largely marked LGG (mutated in 42 of


44 sensitive LGG; Figure 5C, right panel). DAVID analysis of the top 300 differentially


expressed genes highlighted differential mechanisms between solid and non-solid tumors,


such as extracellular matrix and cell motion (Table 5). Altogether, the pharmacogenomics


analyses revealed well-known resistance mechanisms of docetaxel and shed light on the


potential of CX-5461 on hematopoietic malignancies and LGG.


**Limitations and future work**


DNN is unquestionably one of the hottest computational breakthroughs in the era of big


data. Although promising results of our and other studies have demonstrated its ability of


solving challenging bioinformatic tasks, the method has several fundamental limitations.


For instance, due to high representational power and model complexity, the method suffers


                         - 17 

from overfitting and the requirement of large training data. Addressing this, the present


study adopts a training–validation partition of training data to allow early stopping to the


training process [38]. Future work may further incorporate dropout and regularization to


DNNs. Also, by taking advantage of the transferability of neural networks, we used the


huge volume of TGCA data to equip our model the ability of capturing representations of


mutation and expression data. Transferring the learned parameters to initialize our model


virtually increased the sample size of our training data. Our data from 100 iterations of


model training suggest the stability of performance and insensitivity to the selection of


training samples. As the availability of more large-scale drug screening data, we expect the


proposed model to make even more accurate predictions and unveil subtle


pharmacogenomics features. Furthermore, our model may incorporate additional genomic


mutation information, such as copy number alterations, into data matrices @ [8"9:] and


@ [""#!], to enrich the complexity of tumor mutation for model training and further reduce


the training MSE. Because of the nature of DNNs as black boxes, the interpretability of


results is typically limited. In this study, by integrating genomics profiles to the predictions,


we systematically investigated how single gene mutations, as well as the interplay between


cancer type, mutations, and biological functions, were associated with the predicted drug


response. With the advances in DNN, several novel methods were recently proposed to


extract features learned by neural networks, such as network-centric approach [39] and


decomposition of predicted outputs by backpropagation onto specific input features [40]


(reviewed in [41]). Future works may incorporate these methods to provide a landscape of


pharmacogenomics and further reveal novel oncogenic genomics profiles.


                         - 18 

## **Conclusions**

This study addresses the need for a translation of pharmacogenomics features identified


from pre-clinical cell line models to predict drug response of tumors. We developed a DNN


model capable of extracting representative features of mutations and gene expression, and


bridging knowledge learned from cancer cell lines and transferring to tumors. We showed


the reliability of the model and its superior performance than four different methods.


Applying our model to the TCGA collection of tumors, we identified both well-studied and


novel resistance mechanisms and drug targets. Overall, the proposed model is widely


applicable to incorporate other omics data and to study a wider range of drugs, paving the


way to the realization of precision oncology.


                         - 19 

## **List of Abbreviations**

ACC, adrenocortical cancer; BLCA, bladder urothelial carcinoma; BRCA, breast invasive


carcinoma; CCLE, Cancer Cell Line Encyclopedia; CESC, cervical and endocervical


cancer; CHOL, cholangiocarcinoma; COAD, colon adenocarcinoma; DL, deep learning;


DLBC, diffuse large B-cell lymphoma; DNN, deep neural network; E enc, expression


encoder; ER, estrogen receptor; ESCA, esophageal carcinoma; GBM, glioblastoma


multiforme; HNSC, head and neck squamous cell carcinoma; IC 50, half maximal inhibitory


concentration; KICH, kidney chromophobe; KIRC, kidney clear cell carcinoma; KIRP,


kidney papillary cell carcinoma; LAML, acute myeloid leukemia; LGG, lower grade


glioma; LIHC, liver hepatocellular carcinoma; LUAD, lung adenocarcinoma; LUSC, lung


squamous cell carcinoma; MESO, mesothelioma; M enc, mutation encoder; MSE, mean


squared error; MUT, mutated; NSCLC, non-small cell lung cancer; Num, number; OV,


ovarian serous cystadenocarcinoma; P, prediction network; _P_, _P_ -value; PCA, principal


component analysis; PCPG, pheochromocytoma and paraganglioma; PRAD, prostate


adenocarcinoma; Rand Init, random initialization; READ, rectum adenocarcinoma; SARC,


sarcoma; SKCM, skin cutaneous melanoma; STAD, stomach adenocarcinoma; SVM,


support vector machine; TCGA, The Cancer Genome Atlas; TGCT, testicular germ cell


tumor; THCA, thyroid carcinoma; THYM, thymoma; UCEC, uterine corpus endometrioid


carcinoma; UCS, uterine carcinosarcoma; UVM, uveal melanoma; WT, wildtype


                         - 20 

## **Declarations**

**Ethics approval and consent to participate**


Not applicable.


**Consent for publication**


Not applicable.


**Availability of data and material**


The dataset supporting the conclusions of this article is included within the article.


**Competing interests**


The authors declare that they have no competing interests.


**Funding**


This research and this article's publication costs were supported partially by the NCI Cancer


Center Shared Resources (NIH-NCI P30CA54174 to YC), NIH (CTSA 1UL1RR025767

01 to YC, and R01GM113245 to YH), CPRIT (RP160732 to YC), and San Antonio Life


Science Institute (SALSI Innovation Challenge Award 2016 to YH and YC). The funding


sources had no role in the design of the study and collection, analysis, and interpretation of


data and in writing the manuscript.


**Authors' contributions**


YCC, HHC, TZ, SZ, AG, LJW, YH, and YC conceived the study. YCC, YH, and YC


designed the model. YCC performed data analysis. YCC, YH, and YC interpreted the data.


YCC, HHC, TZ, SZ, AG, LJW, YH, and YC wrote and approved the final version of paper.


                         - 21 

**Acknowledgements**


None.




- 22 

## **References**

1. Hanahan D, Weinberg RA: **Hallmarks of cancer: the next generation** . _Cell_
2011, **144** (5):646-674.
2. Schmitt MW, Loeb LA, Salk JJ: **The influence of subclonal resistance**
**mutations on targeted cancer therapy** . _Nat Rev Clin Oncol_ 2016, **13** (6):335347.
3. Phillips KA, Veenstra DL, Oren E, Lee JK, Sadee W: **Potential role of**
**pharmacogenomics in reducing adverse drug reactions: a systematic review** .
_JAMA : the journal of the American Medical Association_ 2001, **286** (18):22702279.
4. Hertz DL, Rae J: **Pharmacogenetics of cancer drugs** . _Annu Rev Med_ 2015,
**66** :65-81.
5. Mina M, Raynaud F, Tavernari D, Battistello E, Sungalee S, Saghafinia S, Laessle
T, Sanchez-Vega F, Schultz N, Oricchio E _et al_ : **Conditional Selection of**
**Genomic Alterations Dictates Cancer Evolution and Oncogenic**
**Dependencies** . _Cancer cell_ 2017, **32** (2):155-168 e156.
6. Park S, Lehner B: **Cancer type-dependent genetic interactions between cancer**
**driver alterations indicate plasticity of epistasis across cell types** . _Molecular_
_systems biology_ 2015, **11** (7):824.
7. Iorio F, Knijnenburg TA, Vis DJ, Bignell GR, Menden MP, Schubert M, Aben N,
Goncalves E, Barthorpe S, Lightfoot H _et al_ : **A Landscape of**
**Pharmacogenomic Interactions in Cancer** . _Cell_ 2016, **166** (3):740-754.
8. Yang W, Soares J, Greninger P, Edelman EJ, Lightfoot H, Forbes S, Bindal N,
Beare D, Smith JA, Thompson IR _et al_ : **Genomics of Drug Sensitivity in**
**Cancer (GDSC): a resource for therapeutic biomarker discovery in cancer**
**cells** . _Nucleic Acids Res_ 2013, **41** (Database issue):D955-961.
9. Alipanahi B, Delong A, Weirauch MT, Frey BJ: **Predicting the sequence**
**specificities of DNA- and RNA-binding proteins by deep learning** . _Nat_
_Biotechnol_ 2015, **33** (8):831-838.
10. Zhou J, Troyanskaya OG: **Predicting effects of noncoding variants with deep**
**learning-based sequence model** . _Nature methods_ 2015, **12** (10):931-934.
11. Lin C, Jain S, Kim H, Bar-Joseph Z: **Using neural networks for reducing the**
**dimensions of single-cell RNA-Seq data** . _Nucleic Acids Res_ 2017, **45** (17):e156.
12. Angermueller C, Lee HJ, Reik W, Stegle O: **DeepCpG: accurate prediction of**
**single-cell DNA methylation states using deep learning** . _Genome biology_ 2017,
**18** (1):67.
13. Chaudhary K, Poirion OB, Lu L, Garmire LX: **Deep Learning-Based Multi-**
**Omics Integration Robustly Predicts Survival in Liver Cancer** . _Clinical_
_cancer research : an official journal of the American Association for Cancer_
_Research_ 2017.
14. Wen M, Zhang Z, Niu S, Sha H, Yang R, Yun Y, Lu H: **Deep-Learning-Based**
**Drug-Target Interaction Prediction** . _Journal of proteome research_ 2017,
**16** (4):1401-1409.
15. Kadurin A, Aliper A, Kazennov A, Mamoshina P, Vanhaelen Q, Khrabrov K,
Zhavoronkov A: **The cornucopia of meaningful leads: Applying deep**


                         - 23 

**adversarial autoencoders for new molecule development in oncology** .
_Oncotarget_ 2017, **8** (7):10883-10890.
16. Preuer K, Lewis RPI, Hochreiter S, Bender A, Bulusu KC, Klambauer G:
**DeepSynergy: Predicting anti-cancer drug synergy with Deep Learning** .
_Bioinformatics_ 2017.
17. Patro R, Duggal G, Love MI, Irizarry RA, Kingsford C: **Salmon provides fast**
**and bias-aware quantification of transcript expression** . _Nature methods_ 2017,
**14** (4):417-419.
18. Newton Y, Novak AM, Swatloski T, McColl DC, Chopra S, Graim K, Weinstein
AS, Baertsch R, Salama SR, Ellrott K _et al_ : **TumorMap: Exploring the**
**Molecular Similarities of Cancer Samples in an Interactive Portal** . _Cancer_
_Res_ 2017, **77** (21):e111-e114.
19. Barretina J, Caponigro G, Stransky N, Venkatesan K, Margolin AA, Kim S,
Wilson CJ, Lehar J, Kryukov GV, Sonkin D _et al_ : **The Cancer Cell Line**
**Encyclopedia enables predictive modelling of anticancer drug sensitivity** .
_Nature_ 2012, **483** (7391):603-607.
20. Cancer Cell Line Encyclopedia C, Genomics of Drug Sensitivity in Cancer C:
**Pharmacogenomic agreement between two cancer cell line data sets** . _Nature_
2015, **528** (7580):84-87.
21. Kowarik A, Templ M: **Imputation with the R Package VIM** . _2016_ 2016,
**74** (7):16.
22. Alfons A, Templ M: **Estimation of Social Exclusion Indicators from Complex**
**Surveys: The R Package laeken** . _2013_ 2013, **54** (15):25.
23. He K, Zhang X, Ren S, Sun J: **Delving deep into rectifiers: Surpassing human-**
**level performance on imagenet classification** . In: _Proceedings of the IEEE_
_international conference on computer vision: 2015_ ; 2015: 1026-1034.
24. Pumperla M: **Keras + Hyperopt: A very simple wrapper for convenient**
**hyperparameter optimization** . In _._ ; 2016.
25. Shetzer Y, Solomon H, Koifman G, Molchadsky A, Horesh S, Rotter V: **The**
**paradigm of mutant p53-expressing cancer stem cells and drug resistance** .
_Carcinogenesis_ 2014, **35** (6):1196-1208.
26. Hientz K, Mohr A, Bhakta-Guha D, Efferth T: **The role of p53 in cancer drug**
**resistance and targeted chemotherapy** . _Oncotarget_ 2017, **8** (5):8921-8946.
27. Wang JB, Dong DF, Wang MD, Gao K: **IDH1 overexpression induced**
**chemotherapy resistance and IDH1 mutation enhanced chemotherapy**
**sensitivity in Glioma cells in vitro and in vivo** . _Asian Pac J Cancer Prev_ 2014,
**15** (1):427-432.
28. Bywater MJ, Poortinga G, Sanij E, Hein N, Peck A, Cullinane C, Wall M, Cluse
L, Drygin D, Anderes K _et al_ : **Inhibition of RNA polymerase I as a therapeutic**
**strategy to promote cancer-specific activation of p53** . _Cancer Cell_ 2012,
**22** (1):51-65.
29. Hein N, Cameron DP, Hannan KM, Nguyen NN, Fong CY, Sornkom J, Wall M,
Pavy M, Cullinane C, Diesch J _et al_ : **Inhibition of Pol I transcription treats**
**murine and human AML by targeting the leukemia-initiating cell**
**population** . _Blood_ 2017, **129** (21):2882-2895.


                         - 24 

30. Wei JC, Meng FD, Qu K, Wang ZX, Wu QF, Zhang LQ, Pang Q, Liu C:
**Sorafenib inhibits proliferation and invasion of human hepatocellular**
**carcinoma cells via up-regulation of p53 and suppressing FoxM1** . _Acta_
_Pharmacol Sin_ 2015, **36** (2):241-251.
31. Ling X, Calinski D, Chanan-Khan AA, Zhou M, Li F: **Cancer cell sensitivity to**
**bortezomib is associated with survivin expression and p53 status but not**
**cancer cell types** . _J Exp Clin Cancer Res_ 2010, **29** :8.
32. Li P, Zhao M, Parris AB, Feng X, Yang X: **p53 is required for metformin-**
**induced growth inhibition, senescence and apoptosis in breast cancer cells** .
_Biochem Biophys Res Commun_ 2015, **464** (4):1267-1274.
33. Huang da W, Sherman BT, Lempicki RA: **Systematic and integrative analysis**
**of large gene lists using DAVID bioinformatics resources** . _Nature protocols_
2009, **4** (1):44-57.
34. Huang da W, Sherman BT, Lempicki RA: **Bioinformatics enrichment tools:**
**paths toward the comprehensive functional analysis of large gene lists** .
_Nucleic acids research_ 2009, **37** (1):1-13.
35. Fulton B, Spencer CM: **Docetaxel. A review of its pharmacodynamic and**
**pharmacokinetic properties and therapeutic efficacy in the management of**
**metastatic breast cancer** . _Drugs_ 1996, **51** (6):1075-1092.
36. Drygin D, Lin A, Bliesath J, Ho CB, O'Brien SE, Proffitt C, Omori M, Haddach
M, Schwaebe MK, Siddiqui-Jain A _et al_ : **Targeting RNA polymerase I with an**
**oral small molecule CX-5461 inhibits ribosomal RNA synthesis and solid**
**tumor growth** . _Cancer research_ 2011, **71** (4):1418-1430.
37. Xu H, Di Antonio M, McKinney S, Mathew V, Ho B, O'Neil NJ, Santos ND,
Silvester J, Wei V, Garcia J _et al_ : **CX-5461 is a DNA G-quadruplex stabilizer**
**with selective lethality in BRCA1/2 deficient tumours** . _Nature communications_
2017, **8** :14432.
38. Angermueller C, Parnamaa T, Parts L, Stegle O: **Deep learning for**
**computational biology** . _Molecular systems biology_ 2016, **12** (7):878.
39. Yosinski J, Clune J, Nguyen A, Fuchs T, Lipson H: **Understanding neural**
**networks through deep visualization** . _arXiv preprint arXiv:150606579_ 2015.
40. Shrikumar A, Greenside P, Kundaje A: **Learning important features through**
**propagating activation differences** . _arXiv preprint arXiv:170402685_ 2017.
41. Kalinin AA, Higgins GA, Reamaroon N, Soroushmehr S, Allyn-Feuer A, Dinov
ID, Najarian K, Athey BD: **Deep Learning in Pharmacogenomics: From Gene**
**Regulation to Patient Stratification** . _arXiv preprint arXiv:180108570_ 2018.


                         - 25 

## **Tables**

**Table 1 - Performance of our DNNs and other models**


**Measurement** **Model** **Linear** **SVM** **Random** **PCA** **E** **enc** **M** **enc**
**regression** **initialization** **only** **only**



**Median**

**number of**


**training**

**epochs** **[a]**



14 -- -- 9 29 17 9.5



~~a~~ Median of 100 shuffles of training, validation, and testing samples
b Result of one multivariate regression model

c
Results of 265 SVM models, each predicting IC 50 for a drug


                         - 26 

**Table 2 - Top mutations in modulating drug response among individual cancers**



**Num.**

**sensitive**



**Num.**

**resistant**



**Mutation**
**Cancer** **Gene**



**Num.**

**modulated**



**rate**



**drugs**



**drugs** **drugs** **drugs**

**LUAD** _TP53_ 46.1% 235 0 235

**LUSC** _TP53_ 75.1% 228 0 228

**STAD** _TP53_ 43.3% 224 0 224

**HNSC** _TP53_ 66.1% 207 0 207

**COAD** _TP53_ 55.7% 197 0 197

**LIHC** _TP53_ 27.0% 194 1 193

**BRCA** _TP53_ 32.2% 182 7 175

**LGG** _IDH1_ 77.3% 159 138 21

**PRAD** _TP53_ 10.8% 146 1 145

**KIRC** _PBRM1_ 38.0% 142 3 139



**drugs**




- 27 

**Table 3 - Top gene mutations modulating pan-cancer drug response**



**Mutation**
**Gene**



**drugs**



**Num. resistant**



**Num. modulated**



**Num. sensitive**



**rate** **drugs** **drugs** **drugs**

_**TP53**_ 34.3% 251 9 242

_**CSMD3**_ 12.6% 223 12 211

_**SYNE1**_ 11.5% 218 10 208

_**TTN**_ 30.2% 206 44 162

_**RYR2**_ 11.9% 199 14 185

_**USH2A**_ 10.7% 191 12 179

_**LRP1B**_ 12.1% 188 19 169

_**FLG**_ 11.0% 183 9 174

_**MUC16**_ 19.5% 161 51 110

_**PCLO**_ 10.5% 155 12 143

_**PIK3CA**_ 11.7% 144 45 99



**drugs**



**rate**




- 28 

**Table 4 - Top GO clusters enriched in top 300 differentially expressed genes**
**associated with predicted response to docetaxel**


**GO ID** **GO term** **Num. genes** _**P**_ **-value**
**Cluster 1 (enrichment score: 10.89)**
**GO:0007049** cell cycle 40 1.13×10 ~~[-10]~~
**GO:0022402** cell cycle process 33 3.51×10 [-10]
**GO:0000279** M phase 32 1.01×10 [-15]
**Cluster 2 (enrichment score: 3.96)**

nucleotide
**GO:0000166** 56 1.95×10 [-4]

binding



**GO:0032555**



purine
ribonucleotide

binding



54 2.74×10 [-6]



**Cluster 3 (enrichment score: 3.45)**
**GO:0000278** mitotic cell cycle 26 1.01×10 ~~[-9]~~

regulation of
**GO:0007346** 12 3.09×10 [-5]
mitotic cell cycle

**Cluster 4 (enrichment score: 2.47)**

M phase of
**GO:0051327** 8 9.46×10 [-4]
meiotic cell cycle

**GO:0007126** meiosis 8 9.46×10 [-4]
**GO:0051321** meiotic cell cycle 8 1.07×10 [-3]
**Cluster 5 (enrichment score: 2.07)**

chromosome
**GO:0051276** 13 8.64×10 [-2]

organization

mitotic sister



5 2.45×10 [-3]



**GO:0000070**



chromatid
segregation



Each cluster is represented by the largest three GO terms.


                         - 29 

**Table 5 - Top GO clusters enriched in top 300 differentially expressed genes**
**associated with predicted response to CX-5461**


**GO ID** **GO term** **Num. genes** _**P**_ **-value**
**Cluster 1 (enrichment score: 8.65)**

extracellular



**GO:0043062**



17 2.93×10 [-9]


13 2.64×10 [-9]





**GO:0005201**



structure
organization


extracellular

matrix structural


constituent



**Cluster 2 (enrichment score: 6.13)**

epidermis
**GO:0008544** 18 2.35×10 [-9]
development

epithelial cell
**GO:0030855** 8 4.60×10 [-3]
differentiation


**Cluster 3** **(enrichment score: 4.23)**

collagen fibril
**GO:0030199** 9 7.34×10 [-9]

organization

multicellular



**GO:0044259**



organismal
macromolecule
metabolic process



6 8.96×10 [-5]



**Cluster 4 (enrichment score: 2.84)**
**GO:0006928** cell motion 18 8.22×10 [-4]
**GO:0016477** cell migration 13 9.51×10 [-4]
**GO:0048870** cell motility 13 2.33×10 [-3]
**Cluster 5** **(enrichment score: 2.60)**

epithelium
**GO:0060429** 12 6.39×10 [-4]
development

epidermal cell
**GO:0009913** 6 4.49×10 [-3]
differentiation


Each cluster is represented by the largest three GO terms.


                         - 30 

## **Figures**

**Figure 1 - Illustration of the proposed neural network model**


(A) Model overview. Mutation and expression data of TCGA (n = 9,059) were used to pre

train two autoencoders (highlighted in blue and green) to extract data representations.


Encoders of the autoencoders, namely mutation encoder M enc and expression encoder E enc,


were linked to a prediction network (P; denoted in orange) and the entire model ( _i.e._, M enc,


E enc, and P) was trained using CCLE data (n = 622, of which 80%, 10%, and 10% used as


training, validation, and testing, respectively) to predict the response to 265 drugs. (B)


Architecture of the neural networks. Numbers denote the number of neurons at each layer.


**Figure 2 - Model construction and evaluation using CCLE datasets**


(A) Density plots of true (with missing values), imputed, and predicted IC 50 data of CCLE


and predicted data of TCGA. (B) Heatmaps of imputed and predicted IC 50 data of CCLE.


(C, D) Sample-wise Pearson and Spearman correlation between imputed and predicted IC 50


data of CCLE samples. (E) Mean square errors of our and 4 other DNN-based designs. The


proposed model was compared to a model with no TCGA pre-training (with encoders


randomly initialized; abbreviated as Rand Init), with encoders substituted by PCAs, with


E enc only (no M enc ), and with M enc only (no E enc ). Each model was trained for 100 times,


each of which CCLE samples were randomly assigned into training, validation, and testing


sets.


                         - 31 

**Figure 3 - Associations of gene mutations to predicted drug response in TCGA –**


**per-cancer study**


(A) Predicted IC 50 of TCGA tumors with known drug targets in a cancer type. Significance


of ΔIC 50 between tumors with and without a gene mutation was assessed by the two-tailed


_t_ -test. (B) Gene mutations significantly associated with predicted drug response in a cancer


type. Middle panel, significant mutation–drug pairs in each cancer (with Bonferroni


adjusted _t_ -test _P_ < 1.0×10 [-5] ). Nodes labeled with names are those with extreme significance


(adjust _P_ < 1.0×10 [-60] ) and magnitude of ΔIC 50 (|ΔIC 50 | ≥ 0.5). Top 10 cancer types with the


largest sample sizes are denoted by node color and shape. (C) Box plots of three mutation–


drug examples in BRCA and LGG.


**Figure 4 - Associations of gene mutations to predicted drug response in TCGA –**


**pan-cancer study**


(A) Gene mutations significantly associated with predicted drug response across all TCGA


samples. Here only the 11 genes with mutation rates larger than 10% were analyzed. Nodes


labeled with names are those with extreme significance (adjust _P_ < 1.0×10 [-200] ) and


magnitude of ΔIC 50 (ΔIC 50 ≥ 0.7 or ΔIC 50 < 0). (B, C) Examples of drugs modulated by


_TP53_ and _TTN_ mutations, respectively.


**Figure 5 - Pharmacogenomics analysis of docetaxel and CX-5461 in TCGA**


(A) Waterfall plot of predicted IC 50 for the two drugs across all TCGA samples. Tumors


with extreme IC 50 values (top and bottom 1%) were denoted as the resistant and sensitive


groups. (B) Cancer type composition of resistant and sensitive samples. Cancer types


accounted for at least 10% in any group are highlighted in bold and shown in (C). (C)


Heatmaps of cancer type composition, top differentially mutated genes, and top


                         - 32 

differentially expressed genes between the two groups. In the expression heatmap, genes


are normalized and hierarchically clustered, and samples are clustered within each group.


                         - 33 

B



Mutation autoencoder





Inputlayer Encoder

layers



Output

layer



Bottleneck



Decoder

layers









A













|TCGA pretraining|Mutation|
|---|---|
|...<br>...<br>...<br>Input layer<br>Hidden layers<br>...<br>1024<br>256<br>64<br>18281|Prediction<br>network (P)<br>...<br>...<br>...<br>...<br>Merge layer<br>Mutation<br>encoder (Menc)<br>Hidden layers<br>128<br>128<br>128<br>128|
|...<br>...<br>...<br>...<br>1024<br>256<br>64<br>15363|...<br>...<br>...<br>...<br>1024<br>256<br>64<br>15363|
|Input layer<br>Hidden layers|Input layer<br>Hidden layers|


_TCGA_

_pretraining_





Drug

response





Output layer


265







layers Output

layer



Input
layer



layers



Expression autoencoder


# Figure 1


A


C



-10 -5 0 5 10 15
Log IC 50 (µM)


35



B



D E

25



6


5


4


3


2


1


0





CCLE IC 50 CCLE predicted IC 50 Log IC 50 (µM)
Cell lines -8.0 +8.0



4.0


3.5


3.0


2.5


2.0


1.5


1.0





20


15


10


5


0



30


25


20


15


10


5


0







Model Rand Init PCA E enc only M enc only

# Figure 2



0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1

Sample-wise Pearson correlation in log IC 50



0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1

Sample-wise Spearman correlation in log IC 50


A TAMOXIFEN in BRCA B



120


100


80


60


40


20


0



C



TGX221 in LGG





6


5


4


3


2


2


1


0


2


1


0







_TP53_ -wt

(n=274)



ER+
(n=719)



ER(n=214)





_TP53_ -mut

(m=227)



AFATINIB in NSCLC











VINORELBINE in BRCA





_TP53_ -wt

(n=664)



_EGFR_ -mut

(n=56)



_EGFR_ -wt

(n=937)



_TP53_ -mut

(n=316)



GEFITINIB in NSCLC



DOXORUBICIN in LGG



6


5


4


3


-1


-2


-3


-4


-1


-2


-3



_IDH1_ -mut

(n=387)



_IDH1_ -wt
(n=114)



_EGFR_ -mut

(n=56)



_EGFR_ -wt

(n=937)



-2.0 -1.5 -1.0 -0.5 0 0.5 1.0 1.5 2.0


_Sensitive in mut_ _Resistant in mut_


Intra-cancer difference in predicted log IC 50 between mut and wt (µM)


# Figure 3


B
A



CX-5461



SORAFENIB



350


300


250


200


150


100


50



VINORELBINE







_TP53_ DABRAFENIB, GSK690693, GW843682X, IMATINIB, JW-7-24-1,: 5-FLUOROURACIL, BMS-509744, BMS-536924, BX-912, **CX-5461**,

KIN001-236, KIN001-270, LENALIDOMIDE, MG-132, NPK76-II-72-1,


**BORTEZOMIB/**



_**TP53**_













BORTEZOMIB







_TP53_ -wt
(n=5950)



7

6

5

4

3

2

1


13

12

11

10

9

8

7

6

5
4
3



_TP53_ -mut

(n=3109)



_TP53_ -wt
(n=5950)



_TP53_ -mut

(n=3109)



PHENFORMIN



_TP53_ -wt
(n=5950)



_TP53_ -mut

(n=3109)



_TP53_ -wt
(n=5950)



_TP53_ -mut

(n=3109)



C



8

7

6

5

4

3

2

1


8

6

4

2

0


-2

-4

-6


1


0


-1


-2

-3

-4

-5

-6



EPOTHILONE-B


-1

-2

-3

-4

-5

-6



0

-1.5 -1.0 -0.5 0 0.5 1.0 1.5


_Sensitive in mut_ _Resistant in mut_


Pan-cancer difference in predicted log IC 50 between mut and wt (µM)



_TTN_ -mut
(n=2733)



_TTN_ -wt
(n=6326)



_TTN_ -wt
(n=6326)



_TTN_ -mut
(n=2733)


# Figure 4


A


B


C


|Resistant<br>Sensitive|Col2|
|---|---|
|Resistant<br>Sensitive||



Resistant patients Sensitive patients



DOCETAXEL



CX-5461



-1


-2


-3


-4


-5


-6


-7


45

40

35

30

25

20

15

10

5

0











7


6


5


4


3


2


1





1000 2000 3000 4000 5000 6000 7000 8000 9000

Patient



Cancer type











breakdown



Cancer type

breakdown



Resistant patients Sensitive patients

# Figure 5





50

45

40

35

30

25

20

15

10

5

0


**CESC**
**ESCA**
**GBM**
**LGG**
**LIHC**
_CSMD3_
_MUC16_
_MUC4_
_NEB_
_PIK3CA_
_RYR2_
_SYNE1_
_TP53_
_TTN_
_ZFHX4_


-2.0 +2.0



**ESCA**
**HNSC**

**LAML**
**LGG**
**LUSC**
_ADCY1_
_APC_
_CSMD3_
_CUBN_
_FLG_
_IDH1_
_NRXN1_
_PCLO_
_SYNE1_
_TP53_


