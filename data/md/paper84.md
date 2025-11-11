## **Micro-Estimates of Wealth for all Low- and Middle-Income Countries**

Guanghua Chi [1†], Han Fang [2], Sourav Chatterjee [2], Joshua E. Blumenstock [1† ]


**Affiliations:**


1 School of Information, University of California, Berkeley; Berkeley, USA.


2 Facebook, Inc.; Menlo Park, USA.


  - Corresponding author. Email: jblumenstock@berkeley.edu


  - These authors contributed equally to this work.


**Abstract:** Many critical policy decisions, from strategic investments to the allocation of
humanitarian aid, rely on data about the geographic distribution of wealth and poverty. Yet many
poverty maps are out of date or exist only at very coarse levels of granularity. Here we develop
the first micro-estimates of wealth and poverty that cover the populated surface of all 135 low
and middle-income countries (LMICs) at 2.4km resolution. The estimates are built by applying
machine learning algorithms to vast and heterogeneous data from satellites, mobile phone
networks, topographic maps, as well as aggregated and de-identified connectivity data from
Facebook. We train and calibrate the estimates using nationally-representative household survey
data from 56 LMICs, then validate their accuracy using four independent sources of household
survey data from 18 countries. We also provide confidence intervals for each micro-estimate to
facilitate responsible downstream use. These estimates are provided free for public use in the
hope that they enable targeted policy response to the COVID-19 pandemic, provide the
foundation for new insights into the causes and consequences of economic development and
growth, and promote responsible policymaking in support of the Sustainable Development
Goals.


1


Many critical decisions require accurate, quantitative data on the local distribution of wealth and
poverty. Governments and non-profit organizations rely on such data to target humanitarian aid
and design social protection systems ( _1_, _2_ ); businesses use this information to guide their
marketing and investment strategies ( _3_ ); these data also provide the foundation for entire fields
of basic and applied social science research ( _4_ ).


Yet reliable socioeconomic data are expensive to collect, and only half of all countries have
access to adequate data on poverty ( _5_ ). In some cases, the data that do exist are subject to
political capture and censorship ( _6_, _7_ ), and very rarely do such data allow for disaggregation
beyond the largest administrative level ( _8_ ). The scarcity of quantitative data is thus a major
impediment to policymakers and researchers interested in solutions to global poverty and
inequality. Data gaps similarly hinder the broad international coalition working toward the
Sustainable Development Goals, in particular toward the first goal of ending poverty in all its
forms everywhere ( _9_ ).


To address these data gaps, researchers have developed several approaches to construct poverty
maps from non-traditional sources of data. These include methods from small area statistics that
combine household sample surveys with comprehensive census data ( _10_ ), and more recent use of
satellite ‘night-lights’ ( _11_ – _13_ ), mobile phone data ( _14_ ), social media data ( _15_ ), high-resolution
satellite imagery ( _16_ – _19_ ), or some combination of these ( _20_, _21_ ). But these efforts have focused
on a single continent or a select set of countries, limiting their relevance to development
objectives that require a more global perspective.


Here we develop a novel approach to construct micro-regional wealth estimates, and use this
method to create the first complete set of micro-estimates of the distribution of poverty and
wealth across all 135 LMICs (Fig. 1a). We use this method to generate, for each of roughly 19.1
million unique 2.4km micro-regions in all global LMICs, an estimate of the average absolute
wealth (in dollars) and relative wealth (relative to others in the same country) of the people living
in that region. These estimates, which are more granular and comprehensive than previous
approaches, make it possible to see extremely local variation in wealth disparities (Fig. 1b and
Fig. 1c).


Our approach, outlined in Fig. 2, relies on “ground truth” measurements of household wealth
collected through traditional face-to-face surveys with 1,457,315 unique households living in
66,819 villages in 56 different LMICs around the world (Table S1). These Demographic and
Health Surveys (DHS), which are independently funded by the U.S. Agency for International
Development, contain detailed questions about the economic circumstances of each household,
and make it possible to compute a standardized indicator of the average asset-based wealth of
each village (see SM1) ( _8_ ). We then use spatial markers in the survey data to link each village to
a vast array of non-traditional digital data. This includes high-resolution satellite imagery, data
from mobile phone networks, topographic maps, as well as aggregated and de-identified
connectivity data from Facebook (Table S2). These data are processed using deep learning and
other computational algorithms, which convert the raw data to a set of quantitative features of
each village (Fig. S2). We use these features to train a supervised machine learning (ML) model
that predicts the relative wealth (Fig. 1a) and absolute wealth (Fig. S3a) of each all populated
2.4km grid cells in LMICs (see SM2-4).


2


The estimates of wealth and poverty are quite accurate. Depending on the method used to
evaluate performance, the model explains 56-70% of the actual variation in household-level
wealth in LMICs (Fig. 3a). This performance compares favorably to state-of-the-art methods that
focus on single countries or continents ( _16_, _19_ ) (see SM4).


To provide visual intuition for the fine granularity of the wealth estimates, Fig. 1c shows an
enlargement of a region in the outskirts of Cape Town, South Africa. The satellite imagery
shows the physical terrain, which juxtaposes high-density urban areas with farmland and
undeveloped zones by the airport and off the main highway. The bottom half of the figure shows
the wealth estimates for the same region, which highlight the contrast in wealth between these
neighboring areas.


To validate the accuracy of these estimates, and to eliminate the possibility that the ML model is
‘overfit’ on the DHS surveys, we compare the model’s estimates to four independent sources of
ground truth data. The first test uses data from 15 LMICs that have collected and published
census data since 2001 (Table S3). These data contain census survey responses from 27 million
unique individuals, including questions about the economic circumstances of each household.
Importantly, the census data are independently collected and are never used to train the ML
model. In each country, we aggregate the census data at the smallest administrative unit possible
and calculate a ‘census wealth index’ as the average wealth of households in that census unit. We
separately aggregate the 2.4km wealth estimates from the ML model to the same administrative
unit. The ML model explains 72% of the variation in household wealth across the 979 census
units formed by pooling data from the 15 censuses (Fig. 3c) and, on average, 86% of the
variation in household wealth within each of the 15 countries (Fig. S4).


To test the accuracy of the model at the most granular level possible, we obtain three additional
sources of survey data that link household wealth information to the exact geocoordinates of
each surveyed household. The first dataset, collected by the government of the Togolese
Republic (Togo) in 2018-2019, contains a nationally-representative sample of 6,172 households
located in 922 unique 2.4km grid cells (Fig. 4a). We find that the ML model’s predictions
explain 76% of the variation in wealth of these grid cells (Fig. 4b), and 84% of the variation in
wealth of cantons, Togo’s smallest administrative unit (Fig. 4c). The second dataset, similar to
the first but independently collected by the government of Nigeria in 2019, contains a nationallyrepresentative sample of 22,104 households in 2,446 grid cells (Fig. 4d). We find that the ML
estimates explain 50% of the variation in grid cell wealth (Figure 4e) and 71% of the variation in
wealth of Local Government Areas (Fig. 4f).


We further validate the grid-level predictions using a dataset collected by GiveDirectly, a
nonprofit organization that provides humanitarian aid to poor households. In 2018, GiveDirectly
surveyed 5,703 households in two counties in Kenya (Fig. 4g), recording a Poverty Probability
Index as well as the exact geocoordinates of each household (Fig. 4h). Using these data, we show
that even within small rural villages, the ML model’s predictions correlate with GiveDirectly’s
estimates of poverty and wealth (Fig. 4i).


In addition to providing point estimates of the average wealth of the households in each grid cell,
we calculate confidence intervals around each estimate (Fig. S3b). These are obtained through


3


standard resampling methods, combined with a more structural approach that models the
prediction error as a function of observable characteristics of each location (see SM11). As
expected, we find that prediction errors are larger in regions that are far from areas covered by
the DHS data (Table S4). While measures of uncertainty are not common in prior work on subregional wealth estimation, we believe this is an important step to help promote the responsible
use of such estimates in research and policy settings ( _22_ ).


We are making these micro-regional estimates of relative wealth and poverty, along with the
associated confidence intervals, freely available for public use and analysis. These estimates are
provided through an open and interactive data interface that allows scientists and policymakers
to explore and download the data (Fig. S1; see [http://beta.povertymaps.net/ for a preliminary](http://beta.povertymaps.net/)
“beta” version of the interactive interface).


How might these estimates be used to guide real-world policymaking decisions? One key
application is in the targeting of social assistance and humanitarian aid. In the months following
the onset of the COVID-19 pandemic, hundreds of new social protection programs were
launched in LMICs, and in each case, program administrators faced difficult decisions about
whom to prioritize for assistance ( _23_ ). This is because in many LMICs, planners do not have
comprehensive data on the income or consumption of individual households ( _24_ ). The new
estimates provide one potential solution.


In simulations, we find that geographic targeting using our micro-estimates allocates a higher
share of benefits to the poor (and a lower share of benefits to the non-poor) than geographic
targeting approaches based on recent nationally representative household survey data (Table 1
and SM13). This is because the micro-estimates make it possible to target smaller geographic
regions than would be possible with traditional survey data – a finding that is consistent with
prior work that suggests that more granular targeting can produce large gains in welfare ( _2_, _25_,
_26_ ). For instance, the most recent DHS survey in Nigeria only surveyed households in 13.8% of
all Nigerian wards (the smallest administrative unit in the country); by contrast, the microestimates cover 100% of wards. In Togo, existing government surveys only provide poverty
estimates that are representative at the regional level (of which there are only 5); we provide
estimates for 9,770 distinct tiles.


Based on the strength of these results, the Government of Nigeria is using these estimates as the
basis for social protection programs that are providing benefits to millions of poor families ( _27_ ).
Likewise, the Government of Togo is using these estimates to target mobile money transfers to
hundreds of thousands of the country’s poorest mobile subscribers ( _28_ ). These examples
highlight how the ML estimates can improve targeting performance even in countries with robust
national statistical offices, like Nigeria and Togo. In the large number of LMICs that have not
conducted a recent nationally representative household survey, these micro-estimates create an
option for geographic targeting that would otherwise not exist.


The standardized procedure through which these estimates are produced may also be attractive in
contexts where political economy considerations might lead to systematic misreporting of data
( _7_ ) or influence whether new data are collected at all ( _6_ ). However, this does not imply the ML
estimates are apolitical, as maps have a historical tendency to perpetuate existing relations of


4


power ( _29_ ). One particular concern is that the technology used to construct these estimates may
not be transparent to the average user; if not produced or validated by independent bodies, such
opacity might create alternative mechanisms for manipulation and misreporting.


While our primary focus is on constructing, validating, and disseminating this new resource, the
process of building this dataset produces several insights relevant to the construction of highresolution poverty maps. For instance, we find that different sources of input data complement
each other in improving predictive performance ( _20_, _21_ ). While prior work has focused heavily
on satellite imagery, we find that models trained only on satellite data do not perform as well as
models that include other input data (Fig. S7a). In particular, information on mobile connectivity
is highly predictive of sub-regional wealth, with 5 of the 10 most important features in the model
related to connectivity (Fig. S2 and SM5).


The global scale of our analysis also reveals intuitive patterns in the geographic generalizability
of machine learning models ( _16_, _30_, _31_ ). We find that models trained using data in one country
are most accurate when applied to neighboring countries (Fig. S6). Models also perform better in
countries when trained on countries with similar observable characteristics (Table S4). And
while much of the model’s performance derives from being able to differentiate between urban
and rural areas, the model can differentiate variation in wealth within these regions as well (Fig.
3b).


Our hope is that these methods and maps can provide a new set of tools to study economic
development and growth, guide interventions, monitor and evaluate policies, and track the
elimination of poverty worldwide.


5


**References:**
1. J. Blumenstock, Machine learning can help get COVID-19 aid to those who need it most.
_Nature_ . **581** (2020), doi:10.1038/d41586-020-01393-7.
2. C. Elbers, T. Fujii, P. Lanjouw, B. Özler, W. Yin, Poverty alleviation through geographic
targeting: How much does disaggregation help? _Journal of Development Economics_ . **83**,
198–213 (2007).

3. J. N. Sheth, Impact of Emerging Markets on Marketing: Rethinking Existing Perspectives
and Practices. _Journal of Marketing_ . **75**, 166–182 (2011).

4. A. Deaton, Measuring Poverty in a Growing World (or Measuring Growth in a Poor
World). _The Review of Economics and Statistics_ . **87**, 1–19 (2005).

5. U. Serajuddin, C. Wieser, H. Uematsu, A. L. Dabalen, N. Yoshida, “Data deprivation :
another deprivation to end” (WPS7252, The World Bank, 2015), pp. 1–24.

6. R. Pande, F. Blum, Data poverty makes it harder to fix real poverty. That’s why the UN
should push countries to gather and share data. _Washington Post_ (2015), (available at
https://www.washingtonpost.com/news/monkey-cage/wp/2015/07/20/data-poverty-makesit-harder-to-fix-real-poverty-thats-why-the-un-should-push-countries-to-gather-and-sharedata/).

7. J. Sandefur, A. Glassman, The Political Economy of Bad Data: Evidence from African
Survey and Administrative Statistics. _The Journal of Development Studies_ . **51**, 116–132
(2015).

8. S. O. Rutstein, K. Johnson, The DHS Wealth Index. _DHS Comparative Reports_ . **6** (2004).

9. Sustainable Development Solutions Network, “Data for Development: A Needs Assessment
for SDG Monitoring and Statistical Capacity Development” (Sustainable Development
Solutions Network, 2015).

10. C. Elbers, J. O. Lanjouw, P. Lanjouw, Micro–Level Estimation of Poverty and Inequality.

_Econometrica_ . **71**, 355–364 (2003).

11. C. D. Elvidge, P. C. Sutton, T. Ghosh, B. T. Tuttle, K. E. Baugh, B. Bhaduri, E. Bright, A

global poverty map derived from satellite data. _Computers & Geosciences_ . **35**, 1652–1660
(2009).

12. J. V. Henderson, A. Storeygard, D. N. Weil, Measuring Economic Growth from Outer

Space. _American Economic Review_ . **102**, 994–1028 (2012).

13. X. Chen, W. D. Nordhaus, Using luminosity data as a proxy for economic statistics. _PNAS_ .

**108**, 8589–8594 (2011).

14. J. E. Blumenstock, G. Cadamuro, R. On, Predicting poverty and wealth from mobile phone

metadata. _Science_ . **350**, 1073–1076 (2015).

15. I. Weber, R. Kashyap, E. Zagheni, Using Advertising Audience Estimates to Improve

Global Development Statistics. _ITU Journal_ . **1** (2018).

16. N. Jean, M. Burke, M. Xie, W. M. Davis, D. B. Lobell, S. Ermon, Combining satellite

imagery and machine learning to predict poverty. _Science_ . **353**, 790–794 (2016).

17. A. Head, M. Manguin, N. Tran, J. E. Blumenstock, in _Proceedings of the Ninth ACM/IEEE_

_International Conference on Information and Communication Technologies and_
_Development (ICTD 2017)_ (ACM, Lahore, Pakistan, 2017), _ICTD ’17_ .


6


18. R. Engstrom, J. S. Hersh, D. L. Newhouse, “Poverty from space : using high-resolution

satellite imagery for estimating economic well-being” (WPS8284, The World Bank, 2017),
pp. 1–36.

19. C. Yeh, A. Perez, A. Driscoll, G. Azzari, Z. Tang, D. Lobell, S. Ermon, M. Burke, Using

publicly available satellite imagery and deep learning to understand economic well-being in
Africa. _Nature Communications_ . **11**, 2583 (2020).

20. J. E. Steele, P. R. Sundsøy, C. Pezzulo, V. A. Alegana, T. J. Bird, J. E. Blumenstock, J.

Bjelland, K. Engø-Monsen, Y.-A. de Montjoye, A. M. Iqbal, K. N. Hadiuzzaman, X. Lu, E.
Wetter, A. J. Tatem, L. Bengtsson, Mapping poverty using mobile phone and satellite data.
_Journal of The Royal Society Interface_ . **14**, 20160690 (2017).

21. N. Pokhriyal, D. C. Jacques, Combining disparate data sources for improved poverty

prediction and mapping. _PNAS_ . **114**, E9783–E9792 (2017).

22. J. E. Blumenstock, Don’t forget people in the use of big data for development. _Nature_ . **561**,

170–172 (2018).

23. U. Gentilini, M. Almenfi, I. Orton, P. Dale, Social Protection and Jobs Responses to

COVID-19: A Real-Time Review of Country Measures. _World Bank Policy Brief_ (2020)
(available at https://openknowledge.worldbank.org/handle/10986/33635).

24. K. Lindert, T. G. Karippacheril, I. R. Caillava, K. N. Chávez, _Sourcebook on the_

_Foundations of Social Protection Delivery Systems_ (World Bank Publications, 2020).

25. M. Ravallion, Poverty alleviation through regional targeting: a case study for Indonesia.

_The economics of rural organization_, 453–467 (1993).

26. D. P. Coady, The Welfare Returns to Finer Targeting: The Case of The Progresa Program

in Mexico. _Int Tax Public Finan_ . **13**, 217–239 (2006).

27. 1m Nigerians to benefit from COVID-19 Cash Transfer, Osinbajo says. _The Guardian_

(2021), (available at https://guardian.ng/news/1m-nigerians-to-benefit-from-covid-19-cashtransfer-osinbajo-says/).

28. T. Simonite, A Clever Strategy to Distribute Covid Aid—With Satellite Data. _Wired_

(2020), (available at https://www.wired.com/story/clever-strategy-distribute-covid-aidsatellite-data/).

29. J. B. Harley, Maps, knowledge, and power. _Geographic thought: a praxis perspective_, 129–

148 (2009).

30. M. R. Khan, J. E. Blumenstock, in _Proceedings of the 22nd ACM SIGKDD Conference on_

_Knowledge Discovery and Data Mining (KDD 2016)_ (Honolulu, HI, 2016).

31. J. E. Blumenstock, Estimating Economic Characteristics with Phone Data. _American_

_Economic Review: Papers and Proceedings_ . **108**, 72–76 (2018).

32. D. Mahajan, R. Girshick, V. Ramanathan, K. He, M. Paluri, Y. Li, A. Bharambe, L. van der

Maaten, in _Proceedings of the European Conference on Computer Vision (ECCV)_ (2018),
pp. 181–196.

33. M. Ravallion, _The Economics of Poverty: History, Measurement, and Policy_ (Oxford

University Press, New York, 1 edition., 2016).

34. D. Filmer, L. H. Pritchett, Estimating Wealth Effects Without Expenditure Data—Or Tears:

An Application To Educational Enrollments In States Of India*. _Demography_ . **38**, 115–132
(2001).


7


35. A. Deaton, _The Analysis of Household Surveys: A Microeconometric Approach to_

_Development Policy_ (World Bank Publications, 1997).

36. K. He, X. Zhang, S. Ren, J. Sun, in _Proceedings of the IEEE conference on computer vision_

_and pattern recognition_ (2016), pp. 770–778.

37. T. G. Tiecke, X. Liu, A. Zhang, A. Gros, N. Li, G. Yetman, T. Kilic, S. Murray, B.

Blankespoor, E. B. Prydz, H.-A. H. Dang, Mapping the world population one building at a
time. _arXiv:1712.05839 [cs]_ (2017) (available at http://arxiv.org/abs/1712.05839).

38. P. Deville, C. Linard, S. Martin, M. Gilbert, F. R. Stevens, A. E. Gaughan, V. D. Blondel,

A. J. Tatem, Dynamic population mapping using mobile phone data. _PNAS_ . **111**, 15888–
15893 (2014).

39. V. Bahn, B. J. McGill, Testing the predictive performance of distribution models. _Oikos_ .

**122**, 321–331 (2013).

40. L. Breiman, J. Friedman, C. J. Stone, R. A. Olshen, _Classification and Regression Trees_

(Chapman and Hall/CRC, New York, N.Y., 1 edition., 1984).

41. Minnesota Population Center, _Integrated Public Use Microdata Series, International:_

_Version 7.2 [dataset]_ (IPUMS, Minneapolis, MN, 2019;
https://doi.org/10.18128/D020.V7.2).

42. D. J. Hruschka, D. Gerkey, C. Hadley, Estimating the absolute wealth of households.

_Bulletin of the World Health Organization_ . **93**, 483–490 (2015).

43. T. Hellebrandt, P. Mauro, “The Future of Worldwide Income Distribution,” _LIS Working_

_papers_ (635, LIS Cross-National Data Center in Luxembourg, 2015), (available at
https://ideas.repec.org/p/lis/liswps/635.html).

44. M. Jerven, _Poor numbers: how we are misled by African development statistics and what to_

_do about it_ (Cornell University Press, 2013).


8


**Fig. 1 | Micro-estimates of wealth for all low- and middle-income countries** . **a)** Estimates of
the relative wealth of each populated 2.4km gridded region of all 135 LMICs. Interactive version
available at [http://beta.povertymaps.net/. Enlargements show](http://beta.povertymaps.net/) **b)** the countries of South Africa
and Lesotho; **c)** The 12x12km region around the Khayelitsha township near Cape Town, with
0.58m-resolution estimates (both panels show the same region).


9


**Fig. 2 | Overview of approach** . **a)** Nationally-representative household survey data is obtained
from 56 different countries around the world. **b)** In Nigeria, for example, there are 40,680
households surveyed in 899 unique survey locations (“villages”). Non-traditional data from
satellites and other existing sensors are also sourced from each location. **c)** These data are used to
train a machine learning algorithm that predicts micro-regional poverty from non-traditional
data, even in regions where no ground truth data exists.


10


**Fig. 3 | Model performance** . **a)** Distribution of model performance, across 56 countries with
ground truth data, using 3 different approaches to cross-validation. **b)** Much of the model’s
predictive power comes from being able to differentiate between rural and urban locations, but
the model also detects wealth differentials within urban and rural locations. **c)** The ML model
explains 72% of the variation in wealth, as measured with independent census data from 15
LMIC’s. Population-weighted regression lines in blue; 95% confidence intervals in dashes.


11


**Fig. 4 | Validation with independently collected micro-data in Togo, Nigeria, and Kenya** . **a)**
Map of Togo showing locations of surveyed households (jitter added to map to preserve
household privacy). **b)** Scatterplot of the predicted RWI of each grid cell (y-axis) against the
average wealth of the grid cell, as reported in a nationally-representative government survey.
Points sized by population. Population-weighted regression lines in blue; 95% confidence
intervals in shown with dashes. **c)** Scatterplot of predicted RWI against average wealth of each


12


canton, the smallest administrative unit in the country. **d)** Map of surveyed households in Nigeria
(jitter added). **e)** Scatterplot of predicted RWI against average wealth of each grid cell. **f)**
Scatterplot of predicted RWI against average wealth of each local government area (LGA). **g)**
Map of Kenya showing the regions surveyed by GiveDirectly. **h)** Enlargement of the three
survey regions, showing the location of each of 5,703 surveyed households. Colors of
background grid cells indicates RWI predicted from the ML model. In both enlargements, the
width of the grid cell is 2.4km. **i)** Scatterplot of the predicted RWI of each grid cell (y-axis)
against the average PPI of all surveyed households in the grid cell (x-axis). Points are sized by
the number of households in the grid cell, and colored by region.


13


**a)** (1)
**# of spatial**


**units**



(2)
**# of units with**


**estimates**



(3)


_**R**_ _**[2]**_



(4)
**Targeting accuracy,**


**poorest 25%**



(5)
**Targeting accuracy,**


**poorest 50%**



**TOGO**

_Panel A (Togo): High-resolution estimates_


_Panel B (Togo): Imputation based on DHS data_


**NIGERIA**

_Panel C (Nigeria): High-resolution estimates_


_Panel D (Nigeria): Imputation based on DHS data_

**Table 1 | Targeting simulations in Togo and Nigeria. a)** Panels A and C simulate the
performance of anti-poverty programs that geographically targets households using the ML
estimates of tile wealth, under scenarios where the program is implemented at the tile level (first
row) or smallest administrative unit in the country (second row). Panels B and D simulate the
geographic targeting based on the most recent DHS survey, using administrative units of
different sizes. When an admin unit has no surveyed households, the wealth of the unit is
imputed based on the wealth of the geographic unit closest to the household. See Methods and
Table S7 and Table S8 for details. **b)** Map of Togo shows the 47.8% of cantons in Togo in which
DHS household surveys were conducted; unsurveyed cantons shown in grey. **c)** Map of Nigeria
shows surveyed wards in green (13.83% of wards) and unsurveyed wards in grey (86.17%).


14


# Supplementary Materials for

## Micro-Estimates of Wealth for all Low- and Middle-Income Countries

Guanghua Chi, Han Fang, Sourav Chatterjee, Joshua E. Blumenstock


Correspondence to: [jblumenstock@berkeley.edu](mailto:jblumenstock@berkeley.edu)

## **Materials and Methods**


**SM1.** **Ground truth wealth measurements**


The ground truth wealth data used to train the predictive models are derived from household
surveys conducted by the Demographic and Health Survey (DHS) Program. According to the
program, the DHS collects “nationally-representative household surveys that provide data for a
wide range of monitoring and impact evaluation indicators in the areas of population, health, and
nutrition.” [i] We elected to train our model exclusively on DHS data because it is the most
comprehensive single source of publicly available, internationally standardized wealth data that
provides household-level wealth estimates with sub-regional geo-markers.


The fact that we use the DHS data as our ground truth measure of wealth and poverty means that
we are effectively training our machine learning algorithm to reconstruct a DHS-style relative
wealth index – albeit at a much finer spatial resolution and in areas where DHS surveys did not
occur. This is because we believe the DHS version of a relative wealth index is the best publicly
available instrument for consistently measuring wealth across a large number of LMICs.
However, it posits a specific, asset-based definition of wealth that does not necessarily capture a
broader notion of human development. More broadly, a rich social science literature debates the
appropriateness of different measures of human welfare and well-being ( _4_, _33_ ). Our decision to
focus on estimating asset-based wealth, rather than a different measure of socioeconomic status
(SES), was motivated by several considerations. First, in developing economies, where large
portions of the population do not earn formal wages, measures of income are notoriously
unreliable. Instead, researchers and policymakers rely on asset-based wealth indices or measures
of consumption expenditures. Between these two, wealth is much less time-consuming to record
in a survey; as a result, wealth data are more commonly collected in a standardized format for a
large number of countries ( _34_ ).


We obtain the most recent publicly-available DHS survey data from 56 countries (Table S1). The
criteria for inclusion are that the data are available for download through the DHS website (as of
March 2020), the data contain asset/wealth information and sub-regional geomarkers, and that
the most recent survey was conducted since 2000. The combined dataset contains the survey
responses from 1,457,315 household surveys taken across Africa, Asia, Europe, and Latin
America. Each individual household survey lasts several hours, and contains several questions
related to the socioeconomic status of the household. We focus on a standardized set of questions


i http://www.dhsprogram.com/


15


about assets and housing characteristics. [ii] From the responses to these questions, and following
standard practice ( _8_, _35_ ), the DHS calculates a single continuous measure of relative household
wealth, the Relative Wealth Index (RWI), by taking the first principal component of these 15
questions. It is this DHS-computed RWI that we rely upon as a ground truth measure of wealth.


In addition to providing measures of wealth for each household, the DHS indicates the _cluster_ in
which each household is located. The 1.5M households are associated with 66,819 unique
clusters, where a cluster is roughly equivalent to a village in rural areas and a neighborhood in
urban areas. We calculate the average wealth of each “village” cluster by taking the mean RWI
of all surveyed households in that cluster. [iii] This village-level average RWI is the target variable
for the machine learning model.


**SM2.** **Input data**


The prediction algorithms rely on data from several different sources (Table S2). To facilitate
downstream analysis, all data are converted into _features_ that are aggregated at the level of a
2.4km grid cell. We use 2.4km cells because that is the highest resolution at which many of our
input data are available, and it best suited to the spatial merge with the survey data (see
“supervised machine learning” below). We were also concerned that providing estimates of
wealth at even smaller grid cells might compromise the privacy of individual households. Thus,
if the native resolution of a data source is higher than 2.4km, we aggregate the smaller cells to
the 2.4km level by taking the average of the smaller cells.


The features input into the model indicate, for each cell, properties such as the average road
density, the average elevation, and the average annual precipitation. Several features related to
telecommunications connectivity are obtained from Facebook, which uses proprietary methods
to estimate the availability and use of telecommunications infrastructure from de-identified
Facebook usage data [iv] . All estimates are regionally aggregated at the 2.4km level to preserve
user privacy. We use estimates of the number of mobile cellular towers in each grid cell, as well
as the number of WiFi access points and the number of mobile devices of different types. These
measures are based on the infrastructure used by Facebook users, so may not be representative of
the full population. To the extent that these features are predictive of regional wealth (which they
are), no deeper inference or causal interpretation should be drawn from the empirical association.
Rather, these patterns simply indicate that the regional distribution of wealth is correlated with
these non-representative measures of telecommunications use.


Since the raw satellite imagery is extremely high-dimensional, we use unsupervised learning
algorithms to compress the raw data into a set of 100 features. Specifically, following Jean _et al_ .
( _16_ ), we use a pre-trained, 50-layer convolutional neural network to convert each 256x256 pixel
image into 2048 features, and then extract the first 100 principal components of these 2048

ii The full set of indicators are: electricity in household, telephone, automobile, motorcycle, refrigerator, TV, Radio,
water supply, cooking fuel, trash disposal, toilet, floor material, wall material, roof material, and rooms in house.

iii Our main estimates do not use the cluster weights provided by the DHS. We separately evaluate a model that used
these weights to train a weighted regression tree, and find that the predictions of the two models are highly
correlated ( _r_ = 0.9), and result in similar overall performance ( _R_ _[2]_ =0.56 without weights vs. _R_ _[2]_ =0.54 with weights).

iv https://research.fb.com/category/connectivity/


16


dimensional vectors. [v] These 100 components explain 97% of the variance of the 2048 features
(Fig. S8).


All input features are normalized by subtracting the country-specific mean and dividing by the
country-specific standard deviation.


**SM3.** **Spatial join**


We match the ground truth wealth data to the input data using spatial information present in both
datasets. The 2.4km grid cells are defined by absolute latitude and longitude coordinates
specified by the Bing tile system. [vi] The DHS data include approximate information about the
GPS coordinate of the _centroid_ of each of the 66,819 villages. However, the exact
geocoordinates are masked by the DHS program with up to 2km of jitter in urban areas and up to
5km of jitter in rural areas.


To ensure that the input data associated with each village cover the village’s true location, we
include a 2x2 grid of 2.4km cells around the centroid in urban areas, and a 4x4 grid in rural
areas. For each of village, we then take the population-weighted average of the 112-dimensional
feature vectors across 2x2 or 4x4 set of cells, using existing estimates of the population of 2.4km
grid cells ( _37_ ). This leaves us with a training set of 66,819 villages with wealth labels (calculated
from the ground truth data) and 112-dimensional feature vectors (computed from the input data).


**SM4.** **Supervised machine learning**


We use machine learning algorithms to predict the average RWI of each village from the 112
features associated with that village. We do not perform ex ante feature selection prior to fitting
the model. We use a gradient boosted regression tree, a popular and flexible supervised learning
algorithm, to map the inputs to the response variable. To tune the hyperparameters of the
gradient boosted tree, we use three different approaches to cross-validation. [vii]


- _K-fold cross-validation_ (labeled “Basic CV” in Fig. 3a). For each country, the labelled data

are pooled, and then randomly partitioned into _k_ = 5 equal subsets. A model is trained on all
but one subset and tested on the held-out subset. The process is repeated _k_ times and we
report average held-out performance for that country. This approach to cross-validation is
used most frequently in prior work, but can substantially over-estimate performance( _38_ ).


v We use a 50-layer resnet50 network ( _36_ ), where pre-training is similar to Mahajan et. al. ( _32_ ). This network is
trained on 3.5 billion public Instagram images (several orders of magnitude larger than the original Imagenet
dataset) to predict corresponding hashstags. We extract the 2048-dimensional vector from the penultimate layer of
the pre-trained network, without fine-tuning the network weights. The satellite imagery has a native resolution of
0.58 meters/pixel. We downsample these images to 9.375m/pixel resolution by averaging each 16x16 block. The
downsampled images are segmented into 2.4km squares, then passed through the neural network. For each satellite
image, we do a forward-pass through the network to extract the 2048 nodes on the second-to-last layer. We then
apply PCA to this 2048-dimensional object and extract the first 100 components. The PCA eigenvectors are
computed from images in the training dataset (i.e., the images from the 56 countries with household surveys).

vi See https://docs.microsoft.com/en-us/bingmaps/articles/bing-maps-tile-system
vii Hyperparameters were tuned to minimize the cross-validated mean squared error, using a grid search over several
possible values for maximum tree depth (1, 3, 5, 10, 15, 20, 30) and the minimum sum of instance weight needed in
a child (1, 3, 5, 7, 10).


17


This bias arises because both the input (e.g., satellite) and response (RWI) data are spatially
auto-correlated, leaving the training and test data not i.i.d. ( _39_ ). [viii]

- _Leave-one-country-out cross-validation_ (“Leave-country-out”) _._ For each country, a model is

trained using the pooled data from all other 55 countries; the test performance is evaluated on
the held-out country ( _16_ ).

- _Spatially-stratified cross-validation_ (“Spatial CV”) _._ This method ensures that training and

test data are sampled from geographically distinct regions ( _38_, _39_ ). In each country, we select
a random cell as the training centroid, then define the training dataset as the nearest ( _k_ -1)/ _k_
percent of cells to that centroid. The remaining 1/ _k_ cells from that country form the test
dataset. This procedure is repeated _k_ times in each country.


Fig. 3a compares the performance of these three methods, by showing the distribution of _R_ [2]
values for each approach to cross-validation (the distribution is formed from 56 countries, where
a separate model is trained and cross-validated in each country). The difference in _R_ [2] resulting
from different approaches to cross-validation highlights the potential upward bias in performance
that results from spatial auto-correlation in training and test data. By comparison, recent work on
wealth prediction in Africa found that a mixture of remote sensing and nightlight imagery
explains on average 67% of the variation in wealth ( _19_ ). That benchmark was based on an
approach similar to the “leave-country-out” method shown in Fig. 3a; the slight decline in
performance that we observe is likely due to the fact that the 23 countries in Africa studied by
( _19_ ) are substantially more homogenous than the full set of LMICs that we analyze.


Unless noted otherwise, all analysis in this paper uses models based on spatially-stratified crossvalidation. While this has the effect of lowering the _R_ [2] values that we report, we believe it is the
most conservative and appropriate method for training machine learning models on geographic
data with spatial auto-correlation.


**SM5.** **Feature importance**


To shed light on which of the various data sources are driving the model’s predictions, Fig. S2
provides two different indicators of feature importance. Fig. S2a (left panel) indicates the
unconditional correlation between the true wealth label and each individual feature, calculated as
the _R_ _[2 ]_ from a univariate regression of the wealth label on each single feature (each row is a
separate regression; with 56 countries, there are 56 _R_ _[2 ]_ values that form the distribution of each
boxplot). Fig. S2b (right panel) indicates the model gain, which provides an indication of the
relative contribution of each feature to the final model (specifically, it is the average gain across
all splits in the random forest that use that feature)( _40_ ). In general, we find that data related to
connectivity, such as the number of cell towers and mobile devices in a region, are the most
predictive features; nightlight radiance and population density are also predictive. While no
single feature derived from satellite imagery is especially predictive in isolation, the large
number of satellite features collectively contribute to model accuracy – this can be seen most


viii In an extreme example, imagine a single town that covers two adjacent grid cells. If one of the grid cells is in the
training set and the other is in the testing set, a flexible model could simply learn to detect the town and predict its
wealth. This sort of overfitting is not addressed by standard _k_ -fold cross-validation.


18


directly in Fig. S7a, which compares the predictive performance of models with and without
satellite imagery.


**SM6.** **Out-of-sample estimates**


To produce the final maps and micro-estimates, as well as the public dataset, we pool data from
all 56 countries and train a single model using spatially-stratified cross-validation to tune the
model parameters. [ix] This model maps 112-dimensional feature vectors to wealth estimates. We
then pass the 112-dimensional feature vector for each 2.4km grid cell located in a LMIC through
this trained model to produce an estimate of the relative wealth (RWI) of each grid cell (Fig. 1).
We use the World Bank’s List of Country and Lending Groups to define the set of 135 low- and
middle-income countries. [x] Since we do not normalize these predictions at the country level after
they have been generated, we do not expect that each country will have the same within-country
RWI distribution (i.e., the amount of “bright” and “dark” spots will differ between countries).


To help preserve the privacy of individuals and households, we do not display wealth estimates
for 2.4km regions where existing population layers indicate the presence of 50 or fewer
individuals in the region ( _37_ ). Instead, we aggregate neighboring 2.4km tiles (by taking the
population-weighted average RWI) until the total estimated population of the larger area is at
least 50. The “neighbors” of a tile are those tiles that fall within the larger tile, using the tile
boundaries defined by the Bing tile system. [xi] All of the neighboring 2.4km cells in the larger tile
are then assigned the same estimate of RWI (i.e., the population-weighted average).


**SM7.** **Cross-sectional estimation**


Our main objective is to produce accurate estimates of the current, cross-sectional distribution of
wealth and poverty within LMICs. In training the machine learning model described above, we
thus use the most recently available version of each data source. The ground truth wealth
measurements cover a wide range of years (Table S1); the input data are primarily generated in
2018 (Table S2). This often creates a mismatch between the dates of the input variables and the
survey labels for a given region. In practice, this means that our estimates are best at capturing
within-country variation in wealth that does not change over a relatively short time horizon (i.e.,
between the prior survey date and 2018). Analysis of DHS data from LMICs with multiple
surveys suggest a high degree of persistence in the within-country variation in wealth (Fig.
S11) [xii] . Still, this approximation likely introduces error into our model, and suggests that these


ix In robustness analysis, we separately constructed complete micro-estimates for all LMICs in which the estimates
for all countries _without_ DHS surveys were based on the full model trained on pooled data from the 56 countries
with DHS surveys; then, in each of the 56 countries _with_ DHS surveys, we replaced the pooled estimates with the
estimates from a model trained exclusively with data from the target country. We find that the average accuracy of
this alternative approach ( _R_ _[2]_ = 0.54, using spatial CV) is nearly identical to the pooled approach (average _R_ _[2]_ = 0.56,
using spatial CV).

x We use the 2018 version of this list, which includes countries whose Gross National Income per capita was less
than $4,045. See https://datahelpdesk.worldbank.org/knowledgebase/articles/906519-world-bank-country-andlending-groups

xi The 2.4km estimates correspond to Bing tile level 14; the next largest tile, Bing tile level 13, defines 4.8km grid
cells, and so forth.
xii Across the 33 countries with two or more DHS surveys conducted since 2000, the median _R_ _[2]_ between regional
(admin-2) wealth estimates from the most recent DHS survey and the preceding DHS survey is 0.81.


19


estimates are better suited toward applications that require a measure of permanent income than
to applications that require an understanding of poverty dynamics. More broadly, see this
model’s performance as a benchmark that can be improved upon as more input and survey data
become available.


In an ideal world, we would obtain historical input data from the same years in which each
survey was conducted. Unfortunately, historical versions of most of the input data described in
Table S2 do not exist. Alternatively, we could restrict our analysis to input data that do exist in a
historical panel. However, as shown in Fig. S7a, excluding key predictors substantially limits the
model’s predictive accuracy. Another option would be to only train the model using more recent
surveys. In Fig. S9a, we observe that the accuracy of a model trained on the subset of 24
countries that conducted DHS surveys since 2015 is quite similar to the performance of a model
trained on all 56 countries with DHS data since 2000. Related, when we validate the model’s
performance using independently collected census data (see below for details), we find no
evidence to suggest that a shorter gap between the date of the DHS training data and the data of
the census increases the predictive accuracy of the model (Fig. S10).


**SM8.** **Independent validation with census data**


We validate the accuracy of the ML estimates using census data that are collected independently
from the DHS data used to train the models. Specifically, we obtain census data from all
countries with public IPUMS-I data, where the census occurred since 2000 and where asset data
are complete ( _41_ ). In total, these data cover 15 countries on 3 continents, and capture the survey
responses of 27 million individuals (Table S3). We assign each of these individuals a census
wealth index by taking the first principal component of the 13 assets present in the census data.
This list is similar to the DHS asset list, but excludes data on motorcycles and rooms in the
household. As with the DHS data, the PCA eigenvectors are computed separately for each
country. Finally, we compute the average census wealth index over all households within each
second administrative unit, the smallest unit that is consistently available across countries. Of the
1,003 census units, 979 have households with wealth information and also contain a 2.4km tile
with a centroid inside the unit.


Fig. 3c shows a scatterplot of these 979 administrative units, sized by population. The x-axis
indicates the average wealth of each administrative unit, according to the census (calculated as
the mean first principal component across all households in the unit). The y-axis indicates the
average predicted RWI of the administrative unit, calculated by taking the population-weighted
mean RWI of all grid cells within the unit. The population-weighted regression line _R_ _[2]_ = 0.72
(obtained when pooling the 979 admin-2 regions from all 15 countries). Fig. S4 disaggregates
Fig. 3c by country, showing the relationship between census-based wealth and RWI across the
administrative units of each country. The average population-weighted _R_ _[2]_ across the 15 countries
is 0.86 (Table S3);


20


**SM9.** **High-resolution validation with independently collected micro-data from**
**Togo, Nigeria, and Kenya**


We further validate the accuracy of the ML estimates at the finest possible spatial resolution by
comparing them to three independently-collected household surveys in Togo, Nigeria, and
Kenya. In each case, we obtain the original survey data for all households, as well as the exact
GPS coordinates of each surveyed household. As with the census data, none of these datasets
were used to train the ML model; they thus provide an independent and objective assessment of
the accuracy and validity of our new estimates.


_Togo_ . As part of the 2018-2019 Enquete Harmonisee sur les Conditions de Vie des Menages
(EHCVM), the government of Togo conducted a nationally-representative household survey
with 6,172 households. [xiii] A key advantage of these data is that, in addition to observing a wealth
index for each household (calculated as the first principal component of roughly 20 asset-related
questions), we observe each household’s exact geo-coordinates (Fig. 4a). The 6,172 households
are located in 922 unique 2.4km grid cells (which correspond to 260 unique cantons, the smallest
administrative unit in Togo), of the 9,770 total grid cells in the country. We also note that there is
nothing Togo-specific in how the ML model is trained: we simply use the estimates generated by
the final model that is trained using spatially-stratified cross-validation from all 56 countries with
DHS data (also shown in Fig. 1).


Fig. 4a shows the approximate location of each of the households surveyed in the EHCVM. Fig.
4b compares, for each of the 922 grid cells with surveyed households, the average wealth of all
households in each grid cell as calculated from the EHCVM (x-axis) to the estimated RWI of the
grid cell, which is displayed on the y-axis ( _R_ _[2]_ = 0.76). Fig. 4c presents an analogous analysis for
each of the 260 cantons in Togo, where the x-axis indicates the average EHCVM wealth of all
households in the canton and the y-axis indicates the average RWI for each canton, calculated as
population-weighted mean of all cells within the canton ( _R_ _[2]_ = 0.84).


_Nigeria_ . During the 2018-2019 Nigerian Living Standards Survey (NLSS), Nigeria’s National
Bureau of Statistics, in collaboration with the World Bank, conducted a nationally-representative
household survey with 22,104 households (Fig. 4d). [xiv] Like the EHCVM in Togo, the NLSS in
Nigeria contains a wealth index for each household and each household’s exact geo-coordinates.
The 22,104 households are located in 2,446 unique 2.4km grid cells. We compare the NLSS
micro-data, which were never used to train the model, to the final estimates of the ML.


Fig. 4d shows the approximate location of each of the households surveyed in the NLSS. Fig. 4e
compares, for each of the 2,446 grid cells with surveyed households, the average wealth of all
households in the grid cell as calculated from the NLSS to the estimated RWI of the grid cell ( _R_ _[2]_
= 0.50). Fig. 4f presents an analogous analysis for each of the 774 Local Government Areas
(LGAs) in Nigeria ( _R_ _[2]_ = 0.71).


_Kenya_ . We also validate the accuracy of the grid-cell RWI estimates using GPS-enabled survey
data collected in the Kenyan counties of Kilifi and Bomet (Fig. 4g). These data were collected by


xiii See https://inseed.tg/

xiv Borno State was excluded for security concerns. See https://www.nigerianstat.gov.ng/nada/index.php/catalog/64


21


GiveDirectly, a nonprofit organization that provides unconditional cash transfers to poor
households in East Africa. [xv] When GiveDirectly works in a village, they conduct a
socioeconomic survey with every household in the village. The survey includes a standardized
set of 10 questions that form the basis for a Poverty Probability Index (PPI) [xvi], which
GiveDirectly uses to determine which households are eligible to receive cash transfers.
GiveDirectly also records the exact geocoordinates of each household that they survey (Fig. 4g).


Fig. 4i compares estimates of micro-regional wealth based on GiveDirectly’s household PPI
census to corresponding estimates of wealth based on the ML model. We calculate the average
PPI score of each 2.4km grid cell by taking the mean of the PPI scores of all households in the
grid cell. We compare this to the predicted RWI from the ML model. Across the 44 grid cells
shown in Fig. 4h (10 from region 1; 26 from region 2; 8 from region 3), the predicted RWI
explains 21% of the variation in PPI (Pearson’s _r_ = 0.46). Within each region, the correlation
between PPI and RWI ranges from 0.41 – 0.78.


While the ML model explains less of the variation in Kenya than it does in Togo, Nigeria, or in
the 15 census countries, this is a much more stringent test. This is because the comparison is
being done across 44 spatially proximate units (Figure 4h) in 3 small and relatively homogenous
villages. Within these villages, there is less variation in wealth than there is across an entire
country (the variance in RWI across the 44 cells is 0.05; across all of Kenya the variance is
0.10). Our other tests - and indeed all prior work of which we are aware – measure _R_ [2] across
entire countries. The Kenya test is also handicapped by the fact that the Kenyan PPI is not
strictly a wealth index, containing questions about education, consumption, and housing
materials. Measures of wealth and poverty are quite sensitive to the measurement instrument
used. [xvii] To our knowledge, this is the first attempt to compare estimates of micro-regional
wealth, based on variation within single villages, to independently-collected household survey
data where the exact location of each surveyed household is known. We therefore find it
encouraging that the predicted RWI roughly separates wealthier from poorer neighborhoods
within these small regions.


**SM10.** **Model accuracy in high-income nations**


The primary intent of the model is to produce estimates of wealth in LMICs, and it is from
LMICs that we source all of the ground truth data used to train the model. For completeness, we
assess the performance of the model’s predictions in high-income nations. This comparison is
imperfect, because high-income nations do not typically collect asset-based wealth indices,
which is what the ML model is trained to estimate. Instead, we compare the Absolute Wealth
Estimates (AWE) of the ML model (see below for details on how these are constructed) to
independently-produced data on regional Gross Domestic Product per capita (GDPpc) from 30
member nations of the Organisation for Economic Co-operation and Development (OECD).


xv http://www.givedirectly.org/
xvi See https://www.povertyindex.org/country/kenya
xvii For instance, Filmer and Pritchett find that, even within a single survey, the Spearman rank correlation between
an asset index and a measure of consumption expenditures ranges from 0.43 (in Pakistan) to 0.64 (in Nepal). [21]


22


These data are collected by the National Statistical Offices of each respective country, through
the network of Delegates participating in the Working Party on Territorial Indicators. [ xviii]


In each country, we obtain the OECD’s estimate of the average GDPpc of each ‘small’ (TL3)
region. [xix] We separately calculate the AWE of each region by taking the population-weighted
average AWE of all 2.4km grid cells in the region. Fig. S5a shows a scatterplot of these 1540
administrative units, sized by population, where the x-axis indicates the OECD-based measure of
wealth of the administrative unit and the y-axis indicates the population-weighted average
predicted AWE of the administrative unit. Fig. S5b shows the accuracy of the model in each of
the 30 countries. The average population-weighted _R_ _[2]_ across the 30 countries is 0.50; the
population-weighted regression line _R_ _[2]_ = 0.59 (obtained when pooling the 1540 regions from all
30 countries). We note that the AWE values are generally larger than the OECD estimates of
GDPpc (the slope of the regression line in Fig. S5a is 1.35). This is likely due to the fact that the
GDPpc estimates used to construct AWE (sourced from the World Bank) are consistently higher
than the GDPpc estimates sourced from the OECD. This comparison is made in Fig. S5c, where
we compare, for the 30 OECD nations, the relationship between the World Bank estimate of
GDPpc and the average regional GDPpc based on OECD data (the slope of the regression line
Fig. S5c is 1.66).


**SM11.** **Confidence intervals and model error**


In many applied settings, it is important to have not just a point estimate of the wealth of a
particular location, but also to have an understanding of the uncertainty associated with each
point estimate. We are encouraged by the fact that we do not find evidence that the model
performs any worse in poorer regions (Fig. S12), as occurs with nightlights data ( _16_ ).


Disaggregating this error, we find that model error is lower when the target country is near to
many countries with ground truth data used to train the model, and when there are many training
observations nearby. This can be seen in Table S4, where we estimate the error of each
individual 2.4km location _l_ by fitting a linear regression of the model’s residual at _l_ (in the
locations with ground truth data) on observable characteristics of _l_ . We selected a broad set of
observable characteristics that include: all of the features used in the predictive model (with the
exception of the imagery-based features); how much “ground truth” training data was available
near the spatial unit (such as the distance to the nearest DHS cluster); and country-level
characteristics (such as average GDP per capita and continent dummy variables). We then
regress the model error, in RWI units, of grid cell _l_ on the _l_ ’s vector of observable characteristics.
We show the correlates of model error in Table S4, column 1.


xviii Data obtained from https://stats.oecd.org/Index.aspx?DataSetCode=PDB_LV. Of the 36 OECD member
countries, 34 provide data on GDPpc. Of these, we exclude Luxembourg and Malta (which have only one and two
geographic units, respectively). Ireland (6 units) and Lithuania (9 units) are also excluded since the spatial units
listed in the GDPpc data do not match the spatial units listed in the corresponding OECD shapefiles. The remaining
30 countries contain 1690 administrative units, of which 1540 have GDPpc information.
xix The OECD’s TL3 regions typically correspond to second-level administrative regions, with the exception of
Australia, Canada and the United States. These TL3 regions are contained in a TL2 region, with the exception of the
United States for which the Economic Areas cross the States’ borders.


23


To better understand the sensitivity of these error estimates, we re-estimate the results in column
1 of Table S4 using different subsets of available predictors. Columns 2 and 3 of Table S4
indicate that while the point estimates 𝛽 depend somewhat on the other variables included in the
regression, the qualitative patterns are the same. More importantly, we observe that the actual
error estimates (for any given location _l_ ) are not very sensitive to the variables included in the
model. For instance, Fig. S14 compares the error predicted by the model in column 1 of Table
S4 (x-axis) against the error predicted by the two alternative specifications in columns 2-3. Fig.
S14a shows the correlation between the median error of a country under the original
specification and the median error of a country using a new specification that also includes the
100 satellite imagery features as predictors ( _r_ = 0.770). Fig. S14b shows the correlation between
the median country error under the original model and a model that only includes the set of
features that were not used to estimate RWI ( _r_ = 0.773).


More broadly, Fig. S6 and Fig. S13 indicate that models trained with data from a single country
perform best when applied to countries with similar characteristics. To construct Fig. S13, we
calculate the cosine similarity between all pairs of countries based on the country-level attributes
listed in Table S4. [xx] We then show, for different thresholds of dissimilarity _d_, the average test
error across all countries _c_ when the model is trained on countries at least _d_ dissimilar to _c_ . For
instance, when _d =_ 0.1, the model for each country _c_ is trained only on countries at least distance
0.1 from _c_ .


Our objective in constructing the micro-estimates of model error is to provide policymakers and
other users with a sense of where the model is accurate and where it is not. Fig. S3b provides a
granular map of expected model error. We also provide country-level summary statistics of
model error in Table S5 (i.e., the mean, median, and standard deviation of estimated model error
in each country), to provide policymakers in specific countries with at-a-glance estimates of
model performance.


**SM12.** **Absolute wealth estimates**


The predictive models are trained to estimate the Relative Wealth Index (RWI) of each 2.4km
grid cell. The RWI indicates the wealth of that location relative to other locations within the
same country. However, certain practical applications require a measure of the _absolute_ wealth
of a region which can be more directly compared from one country to another.


To provide a rough estimate of the absolute per capita wealth of each grid cell, we use the
technique proposed by Hruschka et al. (2015)( _42_ ) to convert a country’s relative wealth
distribution to a distribution of per-capita GDP. This method relies on three parameters to define
the shape of the wealth distribution: the mean GDP per capita, as a measure of the central
tendency (𝐺𝐷𝑃 𝑐 ); the Gini coefficient, as a measure of dispersion (𝐺𝑖𝑛𝑖 𝑐 ); and a combination of
the Pareto and log-normal distributions that are used to estimate skewness. Specifically, our
Absolute Wealth Estimate (AWE) of a grid cell _i_ in country _c_ is defined by:


xx Specifically, the features are: area, population, island, landlocked, distance to the closest country with DHS,
number of neighboring countries with DHS, GDP per capita, and Gini coefficient.


24


𝐺𝐷𝑃 𝑐
𝐴𝑊𝐸 𝑖𝑐 =  𝑟𝑎𝑛𝑘 𝑖𝑐 ∗ 1
𝑛 [∑𝐼𝐶𝐷𝐹] 𝑗 [𝑐] [(𝑟𝑎𝑛𝑘] [𝑗] [)]


where 𝑟𝑎𝑛𝑘 𝑖𝑐 is the rank of each grid cell’s RWI (relative to other cells in _c_ ), 𝐺𝐷𝑃 𝑐 is the mean
wealth per capita of _c_, and 𝐼𝐶𝐷𝐹 𝑐 is the inverse cumulative distribution of wealth, which is
parameterized exactly following Hruschka et al. (2015). [xxi] We collect indicators of each
country’s Gini coefficient and mean per capita GDP from the sources listed in Table S6, and use
it to produce the Absolute Wealth Estimates (AWE) shown in Fig. a.


This conversion requires strong parametric assumptions about the national distribution of wealth
based on information about the average wealth and wealth inequality in each country. These
assumptions are not justified in many countries, particularly where Gini estimates are unreliable,
or when the ICDF approximation is a poor fit. Thus, the AWE estimates should be treated with
more caution than the RWI estimates, which were carefully validated with several different
sources of independent survey data.


Fig. S15 shows the global distribution of (predicted) absolute wealth, as derived from the
Relative Wealth Index using the above procedure. The figure compares the predicted wealth
distribution based on our method to the global income distribution in 2013, as independently
estimated by Hellebrandt and Mauro (2015)( _43_ ) using household income surveys for more than a
hundred countries that were collected through the Luxembourg Income Study. As expected, the
average wealth distribution, which is a measure of per capita GDP, is uniformly higher than the
estimated income distribution, which reflects actual family incomes (i.e., total economic output
does not translate directly to better family outcomes).


**SM13.** **Targeting simulations**


To illustrate one practical use case for these micro-estimates, we simulate the scenario in which
an anti-poverty program administrator has a fixed budget to distribute to a country’s population.
Following Ravallion ( _25_ ) and Elbers et al. ( _2_ ), we assume that the program will be
geographically targeted, such that all individuals within targeted regions will receive the same
transfer. Our analysis compares the performance of several different approaches to geographic
targeting in Togo (Table S7) and Nigeria (Table S8), with a subset of these results summarized
in Table 1. Performance is evaluated using recent nationally-representative household survey
data collected in each country (see above for a description of the EHCVM and NLSS datasets
used to evaluate targeting outcomes).


In both Table S7 (for Togo) and Table S8 (for Nigeria), Panel A simulates geographic targeting
using the high-resolution ML estimates. The first row simulates a scenario in which cash is
transferred to households located in the poorest 2.4km tiles of the country; the second row
simulates distribution to the households located in the poorest administrative units of the country
(the canton is the smallest administrative unit in Togo and the ward is the smallest administrative



xxi For the Pareto distribution, 𝐼𝐶𝐷𝐹 𝑐 is the inverse cumulative distribution function with shape parameter

1

𝛼= (1 + 𝐺𝑖𝑛𝑖 𝑐 )/(2 𝐺𝑖𝑛𝑖 𝑐 ), using a threshold of

[1− ~~(~~ 𝛼 ~~[)]~~ []][. Otherwise, ][𝐼𝐶𝐷𝐹] [𝑐] [ is for a log-normal distribution based ]



1

𝛼 ~~[)]~~ []][. Otherwise, ][𝐼𝐶𝐷𝐹] [𝑐] [ is for a log-normal distribution based ]



1



𝐺𝑖𝑛𝑖 𝑐 +1



on a normal distribution with a mean of: 𝐿𝑛(𝐺𝐷𝑃𝑝𝑐 𝑐 )−𝜎 [2] /2, where 𝜎= √2∗𝑝𝑟𝑜𝑏𝑖𝑡 ~~(~~



~~)~~ .
2



25


unit in Nigeria), where the wealth of the administrative unit is calculated as the populationweighted average of the RWI of all tiles in that unit. The first column indicates the number of
unique tiles in each country; the second and third columns simply indicate that every spatial unit
(tile or canton/ward) has a corresponding wealth estimate. Column 4 indicates the number of
spatial units for which ground truth data exist (in the EHCVM or NLSS), and column 5 counts
the number of spatial units for which both ML estimates and ground truth data exist. Column 6
indicates the number of households that exist in those spatial units for which there are both ML
estimates and ground truth data. This set of households is then used to measure the correlation
between the ground truth wealth of each household (i.e., “true wealth”) and the ML estimate of
the wealth of the spatial unit in which that household is located (i.e., “predicted wealth”), which
is reported in Column 7. [xxii] In subsequent columns, we assume that the government has a fixed
budget which allows it to only target 25% or 50% of the population. We consider the “true poor”
to be the 25% or 50% of households in the ground truth survey with the lowest household asset
index. In Panel A, the targeting mechanism we simulate selects the 25% or 50% “predicted poor”
households, where the prediction is based on the ML estimate of wealth assigned to the spatial
unit in which each household is located. In instances where including one additional spatial unit
would imply that more than 25% or 50% of households would receive benefits, households from
that region are randomly selected to ensure that exactly 25% or 50% of households receive
benefits. Columns 8 and 9 report the accuracy of this targeting mechanism; columns 10 and 11
report the precision and recall. [xxiii]


For comparison, Panels B-D simulate alternative geographic targeting approaches that a
policymaker might rely on in the absence of comprehensive household-level data on poverty
status, as is the case in many LMICs ( _44_ ). In these simulations, we assume that the policymaker
does not have access to the ML micro-estimates of RWI or the ground truth data from the
EHCVM/NLSS that is used to evaluate their allocation decisions. Instead, the policymaker
designs a geographic targeting policy based on the most recent DHS survey, which was
conducted in 2018 in Nigeria and 2013-14 in Togo.


In Panel B, each row corresponds to targeting at a different level of geographic aggregation. For
instance, the row labeled “prefecture average” in Panel B of Table S7 assumes that the program
will be targeted at the prefecture level, the 2 [nd] -level administrative region in Togo, such that
either all households in the prefecture will receive benefits or none will. Subsequent rows allow
for targeting at smaller geographic units. The columns in Panel B are organized similarly to
Panel A. Note, however, that it is no longer the case that each spatial unit will necessarily have a
“predicted wealth” value. For instance, in the Canton targeting row of Panel B (Column 2)
indicates that only 185 cantons have one or more surveyed households in the most recent DHS
(i.e., only 47.8% of all cantons). Columns 4-6 are analogous to Panel A. In Column 7, the
“predicted wealth” of each household is the average wealth of all households in that region from
the most recent DHS. In subsequent columns, the targeting mechanism selects the 25% or 50%


xxii This table reports the correlation in wealth at the _household_ level, with one observation per household, using the
household survey weights in the EHCVM/NLSS. This approach is most consistent with the targeting simulations,
which require that the policymaker estimate each household’s wealth. This approach is different than that taken to
construct Fig. 4, which shows correlations at the _tile_ level, with one observation per tile, which is consistent with the
earlier objective of evaluating the accuracy of the ML estimates at the geographic level.
xxiii Precision and Recall are always equal in these targeting simulations because the fixed budget constraint implies
that each additional targeting error creates exactly one new false positive and one new false negative.


26


“predicted poor” households, where the prediction is based on the average wealth of all
households in that region from the most recent DHS.


Panel C simulates targeting in a similar manner to Panel B, with one important difference: In
cases where a geographic unit has no surveyed households in the most recent DHS (e.g., 52.2%
of all prefectures in Togo), we impute the wealth of that geographic unit by taking the average
DHS RWI of all households in the geographic unit closest to the household _i_ . The imputation on
Panel C addresses a fundamental limitation of Panel B, which would otherwise leave
policymakers without a mechanism to determine budget allocation in large regions of the country
where survey data do not exist.


Panel D simulates a “nearest neighbor” approach to targeting, where the wealth of a household _i_
is inferred based on the average wealth of the households in the DHS cluster physically closest to
_i_, irrespective of whether those nearest neighbors are located in the same administrative unit as _i_ .


The targeting simulations highlight three main results. First, the ML estimates allow for
geographic targeting at a level of spatial resolution that would not be possible with traditional
survey-based data. As highlighted in prior work ( _25_, _26_, _2_ ), geographic disaggregation can
produce substantial welfare gains. The gains to disaggregation are quantified in the last several
columns of Table S7 and Table S8, which highlight how targeting at the tile level increases both
precision and recall – i.e., it reduces both errors of exclusion and errors of inclusion – relative to
the other targeting options that provide 100% coverage. [xxiv] In practice, it may be logistically
challenging to deliver benefits to such small geographic units, but recent and ongoing work that
uses mobile money to deliver cash transfers directly to beneficiaries suggests that this type of
approach may soon become feasible ( _1_ ).


Second, even if the delivery of benefits will be based on administrative divisions, we find that
admin-region targeting based on the ML estimates performs at least as well as – and often better
than – admin-region targeting based on recent nationally representative household surveys (i.e.,
the comparison of the last row of Panel A to the last row of Panel C or Panel D). This is because
the ML estimates can be used to construct accurate estimates of the wealth of 100% of

administrative units. By contrast, the DHS only surveyed households in 185 (47.8%) cantons in
Togo, and only 1218 (13.8%) wards in Nigeria. Thus, a geographic targeting approach relying on
the DHS data alone would either require implementation at a larger administrative unit, or would
require some other form of imputation into unsurveyed regions (as is the case in Panels C and D)
– both of which adjustments reduce the effectiveness of geographic targeting.


Third, and echoing previous results, the ML estimates are accurate at estimating household
wealth (column 7 of Table S7 and Table S8), and are at least as accurate as household wealth
estimation based on recent DHS data. In this sense, Table S7 and Table S8 provide a
conservative estimate of the gains from using the ML estimates for geographic targeting. Many
LMICs do not have a recent nationally representative household survey available; for instance,


xxiv In Panel B of Table a, the “canton targeting” approach slightly outperforms the tile-level targeting, but as we
discuss below, the approach described in Panel B could not be used to target the majority of cantons in Togo, since
only 47.8% of cantons contain households that participated in the DHS.


27


only 24 of 135 LMICs have conducted a DHS since 2015. For such countries, these microestimates create options for geographic targeting that might otherwise not exist.


Finally, we note that the above discussion compares universal geographic targeting using the ML
estimates to universal geographic targeting using recent DHS data, such that all individuals in a
targeted region receive uniform benefits. In practice, most real-world programs are more
nuanced, and rely on additional targeting criteria (such as proxy means tests and participatory
wealth rankings) to determine program eligibility. These additional criteria would be expected to
increase the performance of all methods listed in Table S7 and Table S8; we do not simulate
those changes to better highlight the gains from geographic disaggregation.


28


**Fig. S1 | Screenshots of the interactive data visualization** . Each grid cell corresponds to a
2.4km grid cell. Absolute wealth (in dollars) indicated by the height of the grid cell. Relative
wealth (relative to other cells in that country) indicated by colors ranging from blue (poorest) to
red (wealthiest). **a)** Region around the Suez canal **. b)** Region around Hyderabad, India.


29


**Fig. S2 | Which input data are most useful?** Two different measures of the importance of each
input variable in predicting sub-regional wealth. We show the distribution of feature importances
for each feature as a boxplot, where the distribution is obtained from training 56 country-specific
models with 5-fold cross-validation. **a)** The _R_ _[2]_ value from a univariate regression of wealth on
each feature. **b)** Gain measures the total contribution of each feature to the final fitted model.
Details on each variable are provided in Table S2. Box plots indicate median (center line),
interquartile range (shaded box), and 1.5x interquartile range (whiskers).


30


**Fig. S3 | Estimates of absolute wealth and of model error. a)** Absolute wealth estimates
(AWE), measured as the average GDP per capita of households in each grid cell. RWI estimates
are converted to AWE estimates using information about the income distribution of each
country. **b)** Predicted absolute error of each grid cell, based on a regression of absolute model
residual on observable characteristics of each grid cell.


31


**Fig. S4 | Model validation using census data in low- and middle-income countries.** For each
of 15 countries with publicly available census data, we compare the average RWI of each
second-level administrative region, as predicted by the ML model, to the average wealth
captured in the census (see Methods). Each dot represents an administrative region, sized by
population. Blue line indicates population-weighted regression line, with 95% confidence
intervals as dashes. Average _R_ [2] across all models is 0.86.


32


**Fig. S5 | Model validation in high-income nations.** Figures compare the model’s estimates of
wealth to data provided by the OECD for 30 member countries. **a)** The 30 nations contain 1540
unique second-level administrative regions, each of which is represented by a dot that is sized
proportional to the population of the region. Figure shows the OECD’s estimate of per capita
GDP for the region (x-axis) vs. the Absolute Wealth Estimates (AWE) of the region generated by
the ML model. Population-weighted regression line _R_ _[2]_ = 0.59. **b)** We separately calculate, for
each of the 30 OECD countries with available GDP data, the _R_ _[2]_ that results from regressing
predicted AWE on GDPpc, across all admin-2 regions within each country. **c)** The estimate of a
country’s GDPpc from the World Bank, which forms the basis for the AWE estimates, is
generally larger than the average regional GDPpc as reported in the OECD data. Values on axes
represent thousands of US Dollars.


33


**Fig. S6 | Geographic generalizability of wealth predictions** . For each of the 56 countries with
ground truth wealth data, a separate model is trained using data from just that country (the
columns in the above matrix). Those models are then tested on previously unseen data from each
of the countries (the rows in the matrix). Colors indicate the _R_ [2] between the model’s predictions
and ground truth. Models generally perform better on nearby and similar countries. Rows and
columns are ordered using a hierarchical clustering algorithm (UPGMA).


34


**Fig. S7 | Models trained only on satellite data do not perform as well as models that include**
**other input data. a)** The distribution of performance across the 56 LMIC’s, measured using
spatially-stratified cross-validation, is shown as three kernel density plots, one for each subset of
input data. The legend reports the average performance ( _R_ _[2]_ ) in black, and the average
performance using standard cross-validation in red (to facilitate comparison to prior work). **b)**
Scatter-plot shows relationship between the actual wealth index (from survey data) and the
predicted wealth index (output by the model), using all 66,819 labeled survey locations on four
continents (AF=Africa, AM=Americas, AS=Asia, EU=Europe).


**Fig. S8 | Feature engineering from satellite imagery.** To reduce the dimensionality of the raw
satellite imagery, we first use a neural network to extract 2048 features from the data (see
Methods), and then apply principal component analysis (PCA) to the set of 2048 features. Figure
shows the cumulative proportion of variance explained by the first _k_ principal components. Our
final model uses 100 components, which cumulatively explain 97% of the total variance of the
2048-dimensional image features.


35


**Fig. S9 | Model performance when trained using surveys from different periods in time. a)**
Performance using recent data only. The solid lines (labeled “All countries”) reproduce the
analysis of Fig. 3a, and show the distribution of model performance for a model trained on 56
countries with DHS data, using 3 different approaches to model cross-validation. The dashed
lines indicate the performance for a model that is trained on the subset of 24 countries where
DHS data was collected in 2015 or later. **b)** Performance using all available survey waves.
Several countries have conducted multiple DHS surveys since 2000. The figure compares the
main model’s performance (using 56 survey-waves from 56 countries) to the performance of a
model trained and evaluated on all available DHS data (117 survey-waves from 56 countries).


36


**Fig. S10 | Validation accuracy for 11 countries with DHS and census data. a)** Of the 56
countries used to train the model, 11 have publicly available census data with asset information.
The table indicates the dates of the most recent DHS survey and census in each country, as well
as the correspondence ( _R_ _[2]_ ) between the model predictions and the census data (see also Fig. **S4** ).
**b)** Figure illustrates that there is no clear relationship between the gap in years between the most
recent DHS survey and census (x-axis) against the validation accuracy of the model, for each of
these 11 countries.


37


**Fig. S11 | Temporal stability of within-country wealth over time.** For each country with two
or more DHS since 2000, each subfigure plots the relationship between the average RWI of each
2 [nd] -level administrative unit as computed from the most recent DHS (x-axis) against the average
RWI of the same unit as computed from the previous DHS. Each circle represents an
administrative region. Blue line indicates population-weighted regression line, with 95%
confidence intervals as dashes. Median (mean) _R_ [2] across all countries is 0.81 (0.78).


38


**Fig. S12 | Model performance and country characteristics.** a) For each of the 56 countries
with ground truth data from the DHS, the figure plots the country-level _R_ [2] (measured using basic
5-fold cross-validation) against that country’s GDP per capita, as measured in Table S6. **b)**
Coefficients and standard errors from a regression of the country-level _R_ [2 ] on country-level
characteristics, for the 56 countries with ground truth data, indicates that model performance is
slightly worse in upper middle-income countries (relative to the omitted category of lowermiddle income countries, but is not significantly different in low-income countries or specific
continents.


**Fig. S13 | Models perform better on countries with similar characteristics.** X-axis shows the
10 deciles of the cosine dissimilarity distribution (i.e., one minus cosine similarity). Y-axis
indicates the average prediction error across test countries, where a separate model for each test
country is trained using data from countries at least _d_ dissimilar to the test country.


39


**Fig. S14 | Stability of error estimation to different model specifications.** Figure compares the
median error of all grid cells in a country from the base model (defined by column 1 of Table S4,
and plotted on the x-axis of both figures) to two alternative model specifications (plotted on the
y-axis). Each dot represents a country. **a)** Alternate model includes 100 satellite-based features
(defined by column 2 of Table S4). **b)** Alternate model limited to only features not used to
predict RWI (defined by column 3 of Table S4).


**Fig. S15 | The global income and estimated wealth distribution.** Orange line shows the global
income distribution in 2013, based on household income surveys for more than a hundred
countries. Blue line shows the distribution of predicted “absolute wealth”, a measure of per
capita GDP, which is derived from the Relative Wealth Index that is the focus of this paper.


40


**Country** **Code** **Survey Year** **# households** **# villages**


41


|53<br>54<br>55<br>56|Togo TG 2013-14 9,549 330<br>Uganda UG 2016 19,284 685<br>Zambia ZM 2018 12,595 535<br>Zimbabwe ZW 2015 10,534 400|
|---|---|
||**_Total_**<br> <br> <br>**_1,457,315_**<br>**_66,819_**|


**Table S1 | Ground truth data** . The relative wealth prediction model is trained on nationally
representative Demographic and Health Surveys from 56 countries. See www.dhsprogram.com.


42


**Resolution** **Source** **Min** **Mean** **Median** **Max**















**Table S2 | Data Sources** . Summary statistics for the different datasets that are used as input to
the machine learning algorithms. We use the most recently available data layer from each source.
Publicly available data denoted by *; Data requiring license or other restrictions denoted by [+] .

_Sources:_ 1 http://www.openstreetmap.org
2 http://www.landcover.org/data/lc/
3 https://lta.cr.usgs.gov/SRTM1Arc
4 https://disc.gsfc.nasa.gov/
5 https://data.humdata.org/dataset/highresolutionpopulationdensitymaps
6 https://research.fb.com/category/connectivity/

7 https://www.ngdc.noaa.gov/eog
8 http://www.digitalglobe.com/
9 http://www.dhsprogram.com/
10 https://international.ipums.org/international/
11 https://stats.oecd.org/Index.aspx?DataSetCode=PDB_LV
12 http://www.givedirectly.org/


43


**Country** **Survey Year** **# Households** **# Individuals** **# Admin units** _**R**_ _**[2]**_

|1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13<br>14<br>15|Benin 2013 180,621 1,009,693 77 0.90<br>Dominican Republic 2010 268,637 943,784 67 0.90<br>Haiti 2003 179,190 838,045 28 0.96<br>Honduras 2001 123,584 608,620 99 0.93<br>Lesotho 2006 41,726 180,208 64 0.77<br>Mexico 2015 2,927,196 11,344,365 32 0.80<br>Nicaragua 2005 105,629 515,485 70 0.94<br>Panama 2010 95,579 341,118 36 0.90<br>Rwanda 2012 242,461 1,038,369 30 0.82<br>Senegal 2002 107,999 994,562 28 0.95<br>Sierra Leone 2004 82,518 494,298 108 0.83<br>Tanzania 2012 950,776 4,498,022 114 0.82<br>Togo 2010 121,237 584,859 37 0.93<br>Venezuela 2001 543,475 2,306,489 158 0.54<br>Zambia 2010 250,805 1,321,973 55 0.94|
|---|---|
||**_Total_**<br> <br>**_6,221,433_**<br>**_27,019,890_**<br>**_1,003_**<br>**_Avg: 0.86_**|



**Table S3 | Census validation data** . Census data from 27 million individuals in 15 countries
were used to provide independent validation of the wealth estimates. Data obtained from IPUMS
( _41_ ). The final column indicates the proportion of variance in wealth (as measured in the census)
explained by the model’s wealth estimates (RWI) – see Fig. S4.


44


|(1)<br>Base specification<br>𝜷 SE|(2) (3)<br>Incl. imagery Excluding all RWI<br>features features<br>𝜷 SE 𝜷 SE|
|---|---|
|ln(Dist. to closest DHS country)<br>0.0846***<br>0.007<br> <br>ln(Dist. the closest DHS cluster)<br>0.0217**<br>0.01<br> <br>ln(# neighbor countries w/ DHS)<br>0.0124**<br>0.005<br> <br>ln(# DHS clusters w/in 50 km)<br>-0.0115***<br>0.003<br> <br>ln(# DHS clusters w/in 250 km)<br>-0.0131***<br>0.002<br> <br>ln(# DHS clusters w/in 500 km)<br>-0.0004<br>0.001<br> <br>ln(# DHS clusters w/in 1000 km)<br>0.0225***<br>0.002<br> <br>Is island<br>0.0209**<br>0.01<br> <br>Is landlocked<br>-0.0153**<br>0.006<br> <br>Is America<br>-0.1066***<br>0.01<br> <br>Is Asia<br>0.071***<br>0.007<br> <br>Is Europe<br>-0.041***<br>0.015<br> <br>ln(Area)<br>-0.0349***<br>0.004<br> <br>ln(Country population)<br>0.0044*<br>0.003<br> <br>ln(GDP per capita)<br>-0.0151***<br>0.004<br> <br>Gini<br>0.5347***<br>0.04<br> <br>Road density<br>-0.7382*<br>0.382<br> <br>ln(Slope)<br>-0.1535**<br>0.07<br> <br>ln(Elevation)<br>0.0144***<br>0.001<br> <br>ln(Precipitation)<br>0.0166***<br>0.004<br> <br>Is urban or built up<br>-0.0751***<br>0.006<br> <br>ln(Radiance)<br>-0.0033<br>0.004<br> <br>ln(Tile population)<br>0.012***<br>0.002<br> <br>ln(# cell towers)<br>0.0022<br>0.004<br> <br>ln(# Wifi access points)<br>0.0227***<br>0.003<br> <br>ln(# mobile devices)<br>0.1581***<br>0.02<br> <br>ln(# Android devices)<br>-0.1423***<br>0.02<br> <br>ln(# iOS devices)<br>-0.0304***<br>0.003<br> <br>Satellite image features?<br>No<br> <br>Constant<br>-0.225***<br>0.059<br>|<br>0.0922***<br>0.008<br>0.0761***<br>0.007<br> <br>0.0153<br>0.01<br>0.0206**<br>0.01<br> <br>0.0236***<br>0.006<br>0 <br>0.005<br> <br>-0.0084***0.003<br>-0.0158***<br>0.003<br> <br>-0.0144***0.002<br>-0.0106***<br>0.002<br> <br>0.0025<br>0.002<br>-0.0013<br>0.001<br> <br>0.0231***<br>0.002<br>0.023***<br>0.002<br> <br>0.0449***<br>0.011<br>-0.0038<br>0.01<br> <br>-0.0638***0.007<br>-0.0155***<br>0.006<br> <br>-0.1101***0.012<br>-0.0934***<br>0.009<br> <br>-0.0025<br>0.009<br>0.0825***<br>0.007<br> <br>-0.129***<br>0.018<br>-0.0664***<br>0.014<br> <br>-0.0407***0.004<br>-0.034***<br>0.003<br> <br>0.0187***<br>0.003<br>0.011***<br>0.003<br> <br>-0.0313***0.005<br>0.0039<br>0.004<br> <br>0.1347***<br>0.047<br>0.578***<br>0.039<br> <br>0.8917*<br>0.497<br> <br>-0.2316***0.082<br> <br>0.0087***<br>0.002<br> <br>0.0147***<br>0.004<br> <br> <br> <br>-0.0685***0.007<br> <br>-0.0195***0.004<br> <br>0.0112***<br>0.002<br> <br>-0.0015<br>0.004<br> <br>0.0208***<br>0.003<br> <br>0.1696***<br>0.021<br> <br>-0.1493***<br>0.02<br> <br>-0.0308***0.003<br> <br>Yes<br>No<br> <br>-0.0905<br>0.069<br>-0.1614***<br>0.058|



**Table S4 | Correlates of model error** . Table shows coefficients and standard errors from a
regression of model error (defined as the absolute value of the village-level residual from the
predictive model) on a set of characteristics of the village. Three columns correspond to three
different model specifications. Data sources for country-level characteristics are provided in
Table S6. *significant at 10 percent; ** significant at 5 percent; *** significant at 1 percent.


45


**Estimated error** **Mean squared error**

|Col1|Country Code Mean Median S.D. Mean Median S.D.|
|---|---|
|_1 _<br>_2 _<br>_3 _<br>_4 _<br>_5 _<br>_6 _<br>_7 _<br>_8 _<br>_9 _<br>_10 _<br>_11 _<br>_12 _<br>_13 _<br>_14 _<br>_15 _<br>_16 _<br>_17 _<br>_18 _<br>_19 _<br>_20 _<br>_21 _<br>_22 _<br>_23 _<br>_24 _<br>_25 _<br>_26 _<br>_27 _<br>_28 _<br>_29 _<br>_30 _<br>_31 _<br>_32 _<br>_33 _<br>_34 _<br>_35 _<br>_36 _<br>_37 _<br>_38 _<br>_39 _<br>_40 _<br>_41 _<br>_42 _<br>_43 _<br>_44 _<br>_45 _<br>_46 _<br>_47 _<br>_48 _<br>_49 _<br>_50 _|Afghanistan <br>AF <br>0.217 <br>0.213 <br>0.024 <br> <br> <br> <br>Albania <br>AL <br>0.440 <br>0.435 <br>0.061 <br>0.551 <br>0.203 <br>1.257 <br>Algeria <br>DZ <br>0.325 <br>0.308 <br>0.060 <br> <br> <br> <br>American Samoa <br>AS <br>0.647 <br>0.673 <br>0.057 <br> <br> <br> <br>Angola <br>AO <br>0.373 <br>0.368 <br>0.029 <br>0.390 <br>0.227 <br>0.524 <br>Argentina <br>AR <br>0.400 <br>0.394 <br>0.052 <br> <br> <br> <br>Armenia <br>AM <br>0.427 <br>0.408 <br>0.049 <br>0.467 <br>0.242 <br>0.634 <br>Azerbaijan <br>AZ <br>0.260 <br>0.240 <br>0.053 <br> <br> <br> <br>Bangladesh <br>BD <br>0.494 <br>0.503 <br>0.051 <br>0.414 <br>0.264 <br>0.467 <br>Belarus <br>BY <br>0.288 <br>0.273 <br>0.039 <br> <br> <br> <br>Belize <br>BZ <br>0.378 <br>0.359 <br>0.049 <br> <br> <br> <br>Benin <br>BJ <br>0.316 <br>0.305 <br>0.032 <br>0.282 <br>0.113 <br>0.444 <br>Bhutan <br>BT <br>0.378 <br>0.365 <br>0.040 <br> <br> <br> <br>Bolivia <br>BO <br>0.396 <br>0.387 <br>0.035 <br>0.472 <br>0.256 <br>0.610 <br>Bosnia & Herzegovina <br>BA <br>0.360 <br>0.351 <br>0.046 <br> <br> <br> <br>Botswana <br>BW <br>0.393 <br>0.377 <br>0.036 <br> <br> <br> <br>Brazil <br>BR <br>0.395 <br>0.392 <br>0.047 <br> <br> <br> <br>Bulgaria <br>BG <br>0.409 <br>0.407 <br>0.046 <br> <br> <br> <br>Burkina Faso <br>BF <br>0.301 <br>0.294 <br>0.026 <br>0.258 <br>0.062 <br>0.718 <br>Burundi <br>BI <br>0.269 <br>0.259 <br>0.030 <br>0.429 <br>0.169 <br>0.805 <br>Cabo Verde <br>CV <br>0.532 <br>0.528 <br>0.062 <br> <br> <br> <br>Cambodia <br>KH <br>0.503 <br>0.497 <br>0.061 <br>0.387 <br>0.191 <br>0.559 <br>Cameroon <br>CM <br>0.364 <br>0.355 <br>0.033 <br>0.372 <br>0.202 <br>0.445 <br>Central African Republic <br>CF <br>0.382 <br>0.380 <br>0.013 <br> <br> <br> <br>Chad <br>TD <br>0.346 <br>0.348 <br>0.023 <br>0.335 <br>0.067 <br>0.797 <br>China <br>CN <br>0.372 <br>0.349 <br>0.064 <br> <br> <br> <br>Colombia <br>CO <br>0.476 <br>0.465 <br>0.054 <br>0.659 <br>0.209 <br>1.152 <br>Comoros <br>KM <br>0.537 <br>0.510 <br>0.055 <br> <br> <br> <br>Congo <br>CG <br>0.335 <br>0.333 <br>0.013 <br> <br> <br> <br>Congo, Dem. Rep. <br>CD <br>0.378 <br>0.375 <br>0.019 <br>0.307 <br>0.082 <br>0.612 <br>Costa Rica <br>CR <br>0.530 <br>0.537 <br>0.055 <br> <br> <br> <br>Cote d'Ivoire <br>CI <br>0.361 <br>0.348 <br>0.040 <br>0.404 <br>0.210 <br>0.593 <br>Cuba <br>CU <br>0.412 <br>0.398 <br>0.045 <br> <br> <br> <br>Djibouti <br>DJ <br>0.341 <br>0.336 <br>0.024 <br> <br> <br> <br>Dominica <br>DM <br>0.573 <br>0.592 <br>0.057 <br> <br> <br> <br>Dominican Republic <br>DO <br>0.430 <br>0.434 <br>0.049 <br>0.709 <br>0.378 <br>0.890 <br>Ecuador <br>EC <br>0.443 <br>0.429 <br>0.051 <br> <br> <br> <br>Egypt <br>EG <br>0.455 <br>0.476 <br>0.077 <br>0.466 <br>0.252 <br>0.585 <br>El Salvador <br>SV <br>0.417 <br>0.420 <br>0.038 <br> <br> <br> <br>Equatorial Guinea <br>GQ <br>0.335 <br>0.325 <br>0.031 <br> <br> <br> <br>Eritrea <br>ER <br>0.279 <br>0.276 <br>0.012 <br> <br> <br> <br>Eswatini <br>SZ <br>0.469 <br>0.454 <br>0.044 <br>0.466 <br>0.309 <br>0.521 <br>Ethiopia <br>ET <br>0.359 <br>0.360 <br>0.029 <br>0.236 <br>0.075 <br>0.427 <br>Fiji <br>FJ <br>0.502 <br>0.481 <br>0.050 <br> <br> <br> <br>Gabon <br>GA <br>0.390 <br>0.387 <br>0.021 <br>0.347 <br>0.182 <br>0.418 <br>Gambia <br>GM <br>0.291 <br>0.271 <br>0.045 <br> <br> <br> <br>Georgia <br>GE <br>0.334 <br>0.315 <br>0.048 <br> <br> <br> <br>Ghana <br>GH <br>0.348 <br>0.330 <br>0.047 <br>0.363 <br>0.180 <br>0.567 <br>Grenada <br>GD <br>0.583 <br>0.591 <br>0.046 <br> <br> <br> <br>Guatemala <br>GT <br>0.460 <br>0.456 <br>0.054 <br>0.553 <br>0.311 <br>0.704|



46


47


**Table S5 | Estimates of model error in all low and middle-income countries.** Table indicates
mean, median, and standard deviation of predicted model error in all LMICs (columns 3-5). In
countries where ground truth DHS data exist, table reports mean, median, and standard deviation
of mean squared prediction error (columns 6-8).


48


**GDP per**



**Year**

**(GDP)**



**Source (GDP**



**per capita)** **Gini**



**Source**



**Country** **Code** **capita** **(GDP)** **per capita)** **Gini** **(Gini)** **(Gini)**



**Year**

**(Gini)**



**Country** **Code**



**capita**



49


50


**Table S6 | Sources of country-level data** . While most of the country-level statistics come from
the World Bank’s Open Data portal, when the required indicators are missing we use data from
the alternative data sources listed above. Sources below.


51


Submitted Manuscript: Confidential

Template revised February 2021



(1)
**# of**

**spatial**


**units**



(2)
**# of units**


**with**

**estimates**



(3)

**% units**


**with**

**estimates**



(4)
**# units with**


**ground**


**truth**



(5)
**# units with**

**estimates &**

**ground truth**



(6)
**# of households**

**used to evaluate**


**targeting**



(7)


_**R**_ _**[2]**_



(8)
**Targeting**

**accuracy,**

**poorest 25%**



(9)
**Targeting**

**accuracy,**

**poorest 50%**



(10)
**Targeting**
**Precision/Recall**


**poorest 25%**



(11)
**Targeting**
**Precision/Recall**


**poorest 50%**



_Panel A: High-resolution estimates_
Tiles 10,187 10,187 100% 923 923 6,149 0.60 0.73 0.79 0.47 0.79


Canton targeting 387 387 100% 260 260 6149 0.56 0.73 0.77 0.47 0.77


_Panel B: Imputation based on DHS data (not implementable due to incomplete coverage -- locations without DHS data are excluded_

Prefecture targeting 40 40 100% 40 40 6,149 0.49 0.70 0.70 0.39 0.70


Canton targeting 387 185 47.8% 260 149 4,509 0.52 0.76 0.80 0.53 0.80


_Panel C: Imputation based on DHS data (imputing estimates for locations where no DHS data exist)_

Prefecture targeting 40 40 100% 40 40 6,149 0.49 0.70 0.70 0.39 0.70


Canton targeting 387 387 100% 260 260 6,149 0.53 0.72 0.75 0.44 0.75


_Panel D: Imputation based on wealth of k-Nearest neighbors in DHS survey_


Nearest Neighbor - - - - - - 0.57 0.73 0.78 0.45 0.78


Avg of 5 neighbors - - - - - - 0.52 0.70 0.72 0.39 0.72

**Table S7 | Targeting simulations in Togo. a)** Panel A of table simulates the performance of an anti-poverty program that geographically
targets households in the poorest 2.4km tiles in Togo, using the ML estimates of tile wealth. Panels B and C simulate the geographic
targeting of households in the poorest prefectures and cantons of Togo, where the most recent DHS survey is used to estimate the average
wealth of each administrative region. Panel B ignores households regions with no DHS surveys; Panel C assigns such households the
average wealth of the geographic unit closest to the household. Panel D simulates targets poor households where the wealth of a
household is inferred based on the average wealth of the households in the DHS cluster physically closest to it. Simulations are evaluated
using the 2018-19 EHCVM survey.


52


Submitted Manuscript: Confidential

Template revised February 2021



(1)
**# of**

**spatial**


**units**



(2)
**# of units**


**with**

**estimates**



(3)

**% units**


**with**

**estimates**



(4)
**# units with**


**ground**


**truth**



(5)
**# units with**

**estimates &**

**ground truth**



(6)
**# of households**

**used to evaluate**


**targeting**



(7)


_**R**_ _**[2]**_



(8)
**Targeting**

**accuracy,**

**poorest 25%**



(9)
**Targeting**

**accuracy,**

**poorest 50%**



(10)
**Targeting**
**Precision/Recall**


**poorest 25%**



(11)
**Targeting**
**Precision/Recall**


**poorest 50%**



_Panel A: High-resolution estimates_

Tile targeting 159,147 159,147 100% 2,446 2,446 22,060 0.53 0.79 0.79 0.57 0.79

Ward targeting 8,808 8,808 100% 2,016 2,016 22060 0.51 0.78 0.78 0.56 0.78


_Panel B: Imputation based on DHS data (not implementable due to incomplete coverage -- locations without DHS data are excluded)_

State targeting 37 37 100% 37 37 22,060 0.37 0.75 0.74 0.49 0.74

LGA targeting 774 631 81.52% 706 597 19,549 0.47 0.78 0.76 0.55 0.76

Ward targeting 8,808 1,218 13.83% 2,016 464 5,968 0.54 0.83 0.79 0.66 0.79


_Panel C: Imputation based on DHS data (imputing estimates for locations where no DHS data exist)_

State targeting 37 37 100% 37 37 22,060 0.37 0.75 0.74 0.49 0.74

LGA targeting 774 774 100% 706 706 22,060 0.45 0.76 0.75 0.53 0.75

Ward targeting 8,808 8,808 100% 2,016 2,016 22,060 0.46 0.77 0.76 0.54 0.76


_Panel D: Imputation based on wealth of k-Nearest neighbors in DHS survey_


_Nearest Neighbor_ - - - - - 22,060 0.45 0.76 0.76 0.53 0.76

_Avg of 5 neighbors_ - - - - - 22,060 0.46 0.76 0.76 0.52 0.76

**Table S8 | Targeting simulations in Nigeria. a)** Panel A simulates the performance of an anti-poverty program that geographically
targets households in the poorest 2.4km tiles in Nigeria, using the ML estimates of tile wealth. Panels B and C simulate the geographic
targeting of households in the poorest states (admin-2), Local Government Areas (LGAs, admin-3), and wards (admin-4), where the most
recent DHS survey is used to estimate the average wealth of each administrative region. Panel B ignores households regions with no
DHS surveys; Panel C assigns such households the average wealth of the geographic unit closest to the household. Panel D simulates
targets poor households where the wealth of a household is inferred based on the average wealth of the households in the DHS cluster
physically closest to it. Simulations are evaluated using the 2019 NLSS survey.


53


