## **A Marketplace for Data: An Algorithmic Solution**

ANISH AGARWAL, Massachusetts Institute of Technology
MUNTHER DAHLEH, Massachusetts Institute of Technology
TUHIN SARKAR, Massachusetts Institute of Technology


In this work, we aim to design a data marketplace; a robust real-time matching mechanism to efficiently buy
and sell training data for Machine Learning tasks. While the monetization of data and pre-trained models is an
essential focus of industry today, there does not exist a market mechanism to price training data and match
buyers to sellers while still addressing the associated (computational and other) complexity. The challenge in
creating such a market stems from the very nature of data as an asset: (i) it is freely replicable; (ii) its value is
inherently combinatorial due to correlation with signal in other data; (iii) prediction tasks and the value of
accuracy vary widely; (iv) usefulness of training data is difficult to verify a priori without first applying it to a
prediction task. As our main contributions we: (i) propose a mathematical model for a two-sided data market
and formally define the key associated challenges; (ii) construct algorithms for such a market to function and
analyze how they meet the challenges defined. We highlight two technical contributions: (i) a new notion of
“fairness" required for cooperative games with freely replicable goods; (ii) a truthful, zero regret mechanism to
auction a class of combinatorial goods based on utilizing Myerson’s payment function and the Multiplicative
Weights algorithm. These might be of independent interest.


CCS Concepts: • **Theory of computation** → **Algorithmic game theory and mechanism design** .


Additional Key Words and Phrases: Data Marketplaces, Value of Data, Shapley Value, Online Combinatorial

Auctions


**ACM Reference Format:**

Anish Agarwal, Munther Dahleh, and Tuhin Sarkar. 2019. A Marketplace for Data: An Algorithmic Solution.
In _ACM EC ’19: ACM Conference on Economics and Computation (EC ’19), June 24–28, 2019, Phoenix, AZ, USA._
[ACM, New York, NY, USA, 27 pages. https://doi.org/10.1145/3328526.3329589](https://doi.org/10.1145/3328526.3329589)


Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee
provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and
the full citation on the first page. Copyrights for components of this work owned by others than ACM must be honored.
Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires
prior specific permission and/or a fee. Request permissions from permissions@acm.org.
_EC ’19, June 24–28, 2019, Phoenix, AZ, USA_

© 2019 Association for Computing Machinery.
ACM ISBN 978-1-4503-6792-9/19/06...$15.00
[https://doi.org/10.1145/3328526.3329589](https://doi.org/10.1145/3328526.3329589)


**1** **INTRODUCTION**


**A Data Marketplace - Why Now?** Machine Learning (ML) is starting to take the place in industry
that "Information Technology" had in the late 1990s: businesses of all sizes and in all sectors,
are recognizing the necessity to develop predictive capabilities for continued profitability. To be
effective, ML algorithms rely on high-quality training data – however, obtaining relevant training
data can be very difficult for firms to do themselves, especially those early in their path towards
incorporating ML into their operations. This problem is only further exacerbated, as businesses
increasingly need to solve these prediction problems in real-time (e.g. a ride-share company setting
prices, retailers/restaurants sending targeted coupons to clear inventory), which means data gets
“stale" quickly. Therefore, we aim to design a data marketplace – a real-time market structure for
the buying and selling of training data for ML.


**What makes Data a Unique Asset?** (i) Data can be replicated at zero marginal cost – in general,
modeling digital goods (i.e., freely replicated goods) as assets is a relatively new problem (cf. [ 2 ]). (ii)
Its value to a firm is inherently combinatorial i.e., the value of a particular dataset to a firm depends
on what other (potentially correlated) datasets are available - hence, it is not obvious how to set
prices for a collection of datasets with correlated signals. (iii) Prediction tasks and the value of an
increase in prediction accuracy vary widely between different firms - for example, a 10% increase
in prediction accuracy has very different value for a hedge fund maximizing profit compared to
a logistics company trying to decrease inventory costs. (iv) The authenticity and usefulness of
data is difficult to verify a priori without first applying it to a prediction task - continuing the
example from above, a particular dataset of say satellite images may be very predictive for a specific
financial instrument but may have little use in forecasting demand for a logistics company (and
this is infeasible to check beforehand).


**Why Current Online Markets Do Not Suffice?** Arguably, the most relevant real-time markets to
compare against are: (i) online ad auctions (cf. [ 35 ]); (ii) prediction markets (cf. [ 36 ]). Traditionally,
in these markets (e.g. online ad auctions) the commodity (e.g. ad-space) is not a replicable good
and buyers have a strong prior (e.g. historical click-through-rate) on the value of the good sold (cf.

[ 23, 37 ]). In contrast for a data market, _it is infeasible for a firm to make bids on specific datasets as it_
_is unlikely they have a prior on its usefulness_ . Secondly, it is infeasible to run something akin to a
second price auction (and variants thereof) since data is freely replicable (unless a seller artificially
restricts the number of replications, which may be suboptimal for maximizing revenue). This
problem only gets significantly more complicated due to the combinatorial nature of data. _Thus any_
_market which matches prediction tasks and training features on sale, needs to do so based on which_
_datasets collectively are, empirically the most predictive and “cheap" enough for a buyer_ . This is a
capability online ad markets and prediction markets do not currently have. See Section 1.3 for a
more thorough comparison with online ad and prediction markets.


**1.1** **Overview of Contributions**


**Mathematical Model of Two-Sided Data Market. Formal Definition of Key Challenges.** As
the main contribution of this paper, we propose a mathematical model of a system design for a
data marketplace; we rigorously parametrize the participants of our proposed market - the buyers,
the sellers and the marketplace itself (Sections 2.1, 2.2, 2.3) - and the mechanism by which they
interact (Section 2.4). _This is a new formulation, which lays out a possible architecture for a data_
_marketplace, and takes into account some of the key properties that make data unique_ ; it is freely
replicable, it is combinatorial (i.e., features have overlapping information), buyers having no prior
on usefulness of individual datasets on sale and the prediction tasks of buyers vary widely. In
Section 3, we study the key challenges for such a marketplace to robustly function in real-time,


which include: (i) incentivizing buyers to report their internal valuations truthfully; (ii) updating
the price for a collection of correlated datasets such that revenue is maximized over time; (iii)
dividing the generated revenue “fairly" among the training features so sellers get paid for their
marginal contribution; (iv) constructing algorithms that achieve all of the above and are efficiently
computable (e.g. run in polynomial time in the parameters of the marketplace)?


**Algorithmic Solution. Theoretical Guarantees.** In Section 4, we construct algorithms for the
various functions the marketplace must carry out: (i) allocate training features to and collect revenue
from buyers; (ii) update the price at which the features are sold; (iii) distribute revenue amongst the
data sellers. In Section 5, we prove these particular constructions do indeed satisfy the desirable
marketplace properties laid out in Section 3. We highlight two technical contributions: (i) Property
3.4, a novel notion of “fairness" required for cooperative games with freely replicable goods, which
generalizes the standard notion of Shapley fairness; (ii) a truthful, zero regret mechanism for
auctioning a particular class of combinatorial goods based on utilizing Myerson’s payment function
(cf. [ 29 ]) and the Multiplicative Weights algorithm (cf. [ 3 ]). These might be of independent interest.


**1.2** **Motivating Example from Inventory Optimization**

We begin with an example from inventory optimization to help build intuition for our proposed
architecture for a data marketplace (see Section 2 for a mathematical formalization of these dynamics). We refer back to this example throughout the paper as we introduce various notations and
algorithmic constructions.


**Inventory Optimization Example** : Imagine data sellers are retail stores selling anonymized
minute-by-minute foot-traffic data streams into a marketplace and data buyers are logistics companies who want features (i.e. various time series) that best forecast future inventory demand. In
such a setting, even though a logistics company clearly knows there is predictive value in these
data streams on sale, it is reasonable to assume that the company does not have a good prior
on what collection of foot-traffic data streams are most predictive for demand forecasting, and
within their budget. Thus, practically speaking, such a logistics company cannot make accuracy,
individual bids for each data stream (this is even without accounting for the significant additional
complication arising from the overlap in signal i.e., the correlation that invariably will exist between
the foot-traffic data streams of the various retail stores).
Instead what a logistics company does realistically have access to is a well-defined cost model for
not predicting demand well (cf. [ 21, 24 ]) - e.g., “10% over/under-capacity costs $10,000 per week”.
Hence it can make a bid into a data market of what a marginal increase in forecasting accuracy of
inventory demand is worth to it - e.g. “willing to pay $1000 for a percentage increase in demand
forecasting accuracy from the previous week".
In such a setting, the marketplace we design performs the following steps:


(1) The logistics company supplies a prediction task (i.e., a time series of historical inventory
demand) and a bid signifying what a marginal increase in accuracy is worth to it
(2) The market supplies the logistics company with foot-traffic data streams that are “cheap"
enough as a function of the bid made and the current price of the data streams
(3) A ML model is fit using the foot-traffic data streams sold and the historical inventory demand
(4) Revenue is collected based _only on the increased accuracy in forecasting inventory demand_ [1]

(5) Revenue is divided amongst all the retail stores who provided foot-traffic data
(6) The price associated with the foot-traffic data streams is then updated


1 Model evaluation could potentially be done on an out-of-sample test set or based on actual prediction performance on
future unseen demand


What we find especially exciting about this example is that it can easily be adapted to a variety of
commercial settings. Examples include: (i) hedge funds sourcing alternative data to predict certain
financial instruments; (ii) utility companies sourcing electric vehicle charging data to forecast
electricity demand during peak hours; (iii) retailers sourcing online social media data to predict
customer churn.

Thus we believe the _dynamic described above can be a natural, scalable way for businesses to source_
_data for ML tasks, without knowing a priori what combination of data sources will be useful_ .


**1.3** **Literature Review**


**Auction design and Online Matching** . In this work, we are specifically concerned with online
auction design in a two–sided market. There is a rich body of literature on optimal auction design
theory initiated by [ 29 ], [ 31 ]. We highlight some representative papers. In [ 32 ] and [ 10 ], platform
design and the function of general intermediary service providers for such markets is studied;
in [ 17 ], advertising auctions are studied; in the context of ride–sharing such as those in Uber and
Lyft, the efficiency of matching in [ 12 ] and optimal pricing in [ 8 ] are studied. An extensive survey
on online matching, in the context of ad allocation, can be found in [ 27 ]. These paper generally
focus on the tradeoff between inducing participation and extracting rent from both sides. Intrinsic
to such models is the assumption that the value of the goods or service being sold is known partially
or in expectation. This is the key issue in applying these platform designs for a data marketplace; as
stated earlier, it is unrealistic for a buyer to know the value of the various data streams being sold a
priori (recall the inventory example in Section 1.2 in which a logistic company cannot realistically
make accurate bids on separate data streams or bundles of data streams). Secondly these prior
works do no take into account the freely replicable, combinatorial nature of a good such as data.


**Online Ad Auctions** . See [ 35 ] for a detailed overview. There are two key issues with online ad
markets that make it infeasible for data. Firstly, ad-space is not a replicable good i.e., for any
particular user on an online platform, at any instant in time, only a single ad can be shown in an
ad-space. Thus an _online ad market does not need to do any “price discovery"_ - it simply allocates the
ad-space to the highest bidder; to ensure truthfulness, the highest bidder pays the second highest
bid i.e., the celebrated second price auction (and variants thereof). In contrast, for a freely replicable
good such as data, a second price auction does not suffice (unless a seller artificially restricts a
dataset to be replicated a fixed number of times, which may be suboptimal for maximizing revenue).
Secondly, buyers of online ad-space have a strong prior on the value of a particular ad-space - for
example, a pharmaceutical company has access to historical click-through rates (CTR) for when a
user searches for the word “cancer ". So it is possible for firms to make separate bids for different
ad-spaces based on access to past performance information such as CTR (cf. [ 23, 37 ]). In contrast,
since prediction tasks vary so greatly, past success of a specific training feature on sale has little
meaning for a firm trying to source training data for its highly specific ML task; again, making it is
infeasible for a firm to make bids on specific datasets as they have no prior on its usefulness.


**Prediction Markets** . Such markets are a recent phenomenon and have generated a lot of interest,
rightly so. See [ 36 ] for a detailed overview. Typically in such markets, there is a discrete random
variable, _W_, that a firm wants to accurately predict. The market executes as follows: (i) “experts" sell
probability distributions ∆ _W_ i.e., predictions on the chance of each outcome; (ii) the true outcome,
_w_, is observed; (iii) the market pays the “experts" based on ∆ _W_ and _w_ . In such literature, payment
functions based on the Kullback–Leibler divergence are commonly utilized, as they incentivize
“experts" to be truthful (cf. [ 19 ]). Despite similarities, prediction markets remain infeasible for data
as “experts" have to explicitly choose which tasks to make specific predictions for. In contrast,
it is not known a priori whether a particular dataset has any importance for a prediction task;


in the inventory optimization example in Section 1.2, retail stores selling foot-traffic data cannot
realistically know which logistics company’s demand forecast their data stream will be predictive
for (again, this is only exacerbated when taking into account the overlap of information between
features). A data market must instead provide a real-time mechanism to match training features to
prediction tasks based on the increase in predictive value from the allocated features.


**Information Economics** . There has been an exciting recent line of work that directly tackles data
as an economic good which we believe to be complimentary to our work. We divide them into three
major buckets and highlight some representative papers: (i) data sellers have detailed knowledge of
the specific prediction task and incentives to exert effort to collect high-quality data (e.g. reduce
variance) are modeled [ 5, 13 ]; (ii) data sellers have different valuations for privacy and mechanisms
that tradeoff privacy loss vs. revenue gain are modeled [ 16, 22 ]; (iii) studying the profitability of
data intermediaries who supply consumer data to firms that want to sell the very same customers
more targeted goods [ 9 ]. These are all extremely important lines of work to pursue, but they focus
on different (but complementary) objectives.
Referring back to the inventory optimization example in Section 1.2), we model the sellers (retail
stores) as simply trying to maximize revenue by selling foot-traffic data they already collect. Hence
we assume they have _(i) no ability to fundamentally increase the quality of their data stream; (ii) no_
_knowledge of the prediction task; (iii) no concerns for privacy_ . In many practical commercial settings,
these assumptions do suffice as the data is sufficiently anonymized, and these sellers are trying to
monetize data they are already implicitly collecting through their operations. We focus our work
on such a setting, where firms are trying to buy features to feed their ML models, and believe
our formulation to be the most relevant for it. It would be interesting future work to find ways of
incorporating privacy, feedback and the cost of data acquisition into our model.


**2** **THE MODEL - PARTICIPANTS AND DYNAMICS**


**2.1** **Sellers**


Let there be _M_ sellers, each supplying data streams in this marketplace. We formally parameterize
a seller through the following single quantity:


**Feature.** _X_ _j_ ∈ R _[T]_, _j_ ∈[ _M_ ] is a vector of length _T_ .
For simplicity, we associate with each seller a single feature and thus restrict _X_ _j_ to be in R _[T]_ .
Our model is naturally extended to the case where sellers are selling multiple streams of data by
considering each stream as another “seller" in the marketplace. We refer to the matrix denoting
any subset of features as _X_ _S_, _S_ ⊂[ _M_ ] . Recall from the motivations we provide in Sections 1.2
and 1.3 for our model, we assume data sellers do not have the ability to change the quality of the
data stream (e.g. reducing variance) they supply into the market nor any concerns for privacy (we
assume data is sufficiently anonymized as is common in many commercial settings). Additionally
sellers have no knowledge of the prediction tasks their data will be used for and simply aim to
maximize revenue from the datasets that they already have on hand.


**2.2** **Buyers**

Let there be _N_ buyers in the market, each trying to purchase the best collection of datasets they
can afford in this marketplace for a particular prediction task. We formally parameterize a buyer
through the following set of quantities, for _n_ ∈[ _N_ ]:


**Prediction Task.** _Y_ _n_ ∈ R _[T]_ is a vector of _T_ labels that Buyer _n_ wants to predict well [2] .


2 To reduce notational overload, we abstract away the partition of _Y_ _n_ into training and test data.


We provide a clarifying example of _Y_ _n_ and _X_ _j_, using the inventory optimization example in
Section 1.2. There, the historical inventory demand for the logistics company is _Y_ _n_ and each
historical foot-traffic data stream sold by retailers is _X_ _j_ . The “prediction task" is then to forecast
inventory demand, _Y_ _n_, from time-lagged foot traffic data, _X_ _j_ for _j_ ∈[ _M_ ].


**Prediction Gain Function.** G _n_ : R [2] _[T]_ →[ 0, 1 ], the prediction gain function, takes as inputs the
prediction task _Y_ _n_ and an estimate _Y_ [ˆ] _n_, and outputs the quality of the prediction.
For regression, an example of G _n_ is 1 − RMSE [3] (root-mean-squared-error). For classification,
an example of G _n_ is Accuracy [4] . In short, a larger value for G _n_ implies better prediction accuracy.
To simplify the exposition (and without any loss of generality of the model), we assume that all
buyers use the same gain function i.e., G = G _n_ for all _n_ .


**Value of Accuracy.** _µ_ _n_ ∈ R + is how much Buyer _n_ values a _marginal_ increase in accuracy.
As an illustration, recall the inventory optimization example in Section 1.2 where a logistics
company makes a bid of the form, “willing to pay $1000 for a percentage increase in demand
forecasting accuracy from the previous week". We then have the following definition for how a
buyer values an increase in accuracy,


Definition 2.1. _Let_ G _be the prediction gain function. We define the value Buyer_ _n_ _gets from_
_estimate_ _Y_ [ˆ] _n_ _as:_


_µ_ _n_                    - G( _Y_ _n_, _Y_ [ˆ] _n_ )


_i.e., µ_ _n_ _is what a buyer is willing to pay for a unit increase in_ G _._


Remark 2.1. _Though a seemingly natural definition, we view it as one of the key modeling decisions_
_we make in our design of a data marketplace. In particular, a buyer’s valuation for data does not come_
_from specific datasets, but rather from an increase in prediction accuracy of a quantity of interest._


Remark 2.2. _A potential source of confusion is we require_ _µ_ _n_ _to be linear while many ML error_
_metrics are non-linear. For example, in balanced, binary classification problems, randomly guessing_
_labels has expected accuracy of_ 50% _, which has zero value (and not_ _µ_ _n_ / 2 _). However, such non-linearities_
_can easily be captured in the gain function,_ G _. For the balanced, binary classification problem,_ G _can_
_easily be normalized such that_ 50% _-accuracy has value_ 0 _and_ 100% _-accuracy has value 1 (specifically,_
_accuracy_ )
_G_ = _[max]_ [(][0][,] 0 [ �] .5 _)._ _µ_ _n_ _can thus be thought of as a buyer-specific scaling of how much they value an_
_increase in accuracy. Indeed, linear utility models are standard in information economics (c.f. [9])._


Remark 2.3. _Recall that to reduce notational overload, we let_ _Y_ _n_ _refer to both test and train data._
_Specifically,_ _Y_ _n_ = ( _Y_ _n_ _[train]_, _Y_ _n_ _[test]_ ) _. The ML-algorithm accesses_ _Y_ _n_ _[train]_ _and the Gain function,_ G _accesses_
_Y_ _n_ _[test]_ _(i.e._ _Y_ [ˆ] _n_ _), as is standard in ML workflows._


**Public Bid Supplied to Market.** _b_ _n_ ∈ R + is the public bid supplied to the marketplace.
Note that _µ_ _n_ is a private valuation. If Buyer _n_ is strategic, _µ_ _n_ is not necessarily what is revealed
to the marketplace. Thus we define _b_ _n_, which refers to the actual bid supplied to the marketplace
(not necessarily equal to _µ_ _n_ ).



3 RMSE = _Y_ max −1 _Y_ min



� ~~�~~ _Ti_ =1 [(] _[Y]_ [ ˆ] _[i]_ [ −] _[Y]_ _[i]_ [)] [2] [/] _[T]_ [, where: (i)] [ ˆ] _[Y]_ _[i]_ [ is the predicted value for] _[ i]_ [ ∈[] _[T]_ [ ]] [ produced by the machine learning]



algorithm, M, (ii) _Y_ max, _Y_ min are the max and min of _Y_ _n_ respectively.
4 Accuracy = _T_ 1 � _Ti_ =1 [1] [(] _[Y]_ [ ˆ] _[i]_ [ =] _[ Y]_ _[i]_ [)][, with ˆ] _[Y]_ _[i]_ [ defined similarly to that above]


**2.3** **Marketplace**

The function of the marketplace is to match buyers and sellers as defined above. As we make
precise in Section 2.4, we model the _M_ sellers as fixed and the _N_ buyers as coming one at a time.
We formally parameterize a marketplace through the following set of quantities, for _n_ ∈[ _N_ ]:


**Price.** _p_ _n_ ∈ R + is the price the marketplace sets for the features on sale when Buyer _n_ arrives.
As we make precise in Property 3.2, we measure the quality of the prices ( _p_ 1, . . ., _p_ _N_ ) set by
the marketplace for each buyer by comparing against the optimal fixed price in hindsight (i.e.,
standard definition of regret). However, it is well-known that standard price update algorithms
for combinatorial goods, which satisfy Property 3.2, scale very poorly in _M_ (cf. [ 15 ]). Specifically,
if we maintain separate prices for every data stream (i.e., if _p_ _n_ ∈ R + _[M]_ [) it is easily seen that regret-]
minimizing algorithms such as Multiplicative Weights (cf. [ 3 ]) or Upper Confidence Bandits (cf.

[4]), will have exponential running time or exponentially loose guarantees (in _M_ ) respectively. In
fact from [ 15 ], we know regret minimizing algorithms for even very simple non-additive buyer
valuations are provably computationally intractable.
Thus to achieve a zero-regret price update algorithm, without making additional restrictive
assumptions, we restrict _p_ _n_ to be a scalar rather than a _M_ -dimensional vector. This is justified due
to Definition 2.1, where we model a buyer’s “value for accuracy" (and the associated public bid)
through the scalar, _µ_ _n_ (and the scalar _b_ _n_ respectively). This allows the marketplace to control the
quality of the predictions based on the difference between _p_ _n_ and _b_ _n_ (see Section 4.1 for details).


**Machine Learning/Prediction Algorithm.** M : R _[MT]_ → R _[T]_, the learning algorithm utilized by
the marketplace, takes as input the features on sale _X_ _M_, and produces an estimate _Y_ [ˆ] _n_ of Buyer _n_ ’s
prediction problem _Y_ _n_ .
M does not necessarily have to be supplied by the marketplace and is a simplifying assumption.
Instead buyers could provide their own learning algorithm that they intend to use, or point towards
one of the many excellent standard open-source libraries widely used such as SparkML, Tensorflow
and Scikit-Learn (cf. [1, 28, 30]) [5] .


**Allocation Function.** AF : ( _p_ _n_, _b_ _n_ ; _X_ _M_ ) → _X_ [�] _M_, _X_ [�] _M_ ∈ R _[M]_, takes as input the current price _p_ _n_
and the bid _b_ _n_ received, to decide the quality at which Buyer _n_ gets allocated the features on sale
_X_ _M_ (e.g. by adding noise or subsampling the features).
In Section 4.1, we provide explicit instantiations of AF and detailed reasoning for why we
choose this particular class of allocation functions.


**Revenue Function.** RF : ( _p_ _n_, _b_ _n_, _Y_ _n_ ; M, G, _X_ _M_ ) → _r_ _n_, _r_ _n_ ∈ R +, the revenue function, takes as
input the current price _p_ _n_, in addition to the bid and the prediction task provided by the buyer ( _b_ _n_
and _Y_ _n_ respectively), to decide how much revenue _r_ _n_ to extract from the buyer.


**Payment Division Function.** PD : ( _Y_ _n_, _X_ [�] _M_ ; M, G) → _ψ_ _n_, _ψ_ _n_ ∈[ 0, 1 ] _[M]_, the payment-division
function, takes as input the prediction task _Y_ _n_ along with the features that were allocated _X_ [�] _M_, to
compute _ψ_ _n_, a vector denoting the marginal value of each allocated feature for the prediction task.


**Price Update Function.** PF : ( _p_ _n_, _b_ _n_, _Y_ _n_ ; M, G, _X_ _M_ ) → _p_ _n_ +1, _p_ _n_ +1 ∈ R +, the price-update
function, takes as input the current price _p_ _n_, in addition to the bid and the prediction task provided
by the buyer ( _b_ _n_ and _Y_ _n_ respectively) to update the price for Buyer _n_ + 1.


5 Indeed a key trend in many business use cases, is that the ML algorithms used are simply lifted from standard open-source
libraries. Thus the accuracy of predictions is primarily a function of the quality of the data fed to these ML algorithms.


Fig. 1. Overview of marketplace dynamics.


_2.3.1_ _Buyer Utility._ We can now precisely define the utility function, U : R + × R _[T]_ → R, each
buyer is trying to maximize,


Definition 2.2. _The utility Buyer n receives by bidding b_ _n_ _for prediction task Y_ _n_ _is given by_


U( _b_ _n_, _Y_ _n_ ) � _µ_ _n_         - G( _Y_ _n_, _Y_ [ˆ] _n_ ) −RF ( _p_ _n_, _b_ _n_, _Y_ _n_ ) (1)


_where_ _Y_ [ˆ] _n_ = M( _Y_ _n_, _X_ [�] _M_ ) _and_ _X_ [�] _M_ = AF ( _p_ _n_, _b_ _n_ ; _X_ _M_ ) _._


In words, the first term on the right hand side (r.h.s) of (1) is the value derived from a gain in
prediction accuracy (as in Definition 2.1). Note this is a function of the quality of the features that
were allocated based on the bid _b_ _n_ . The second term on the r.h.s of (1) is the amount the buyer
pays, _r_ _n_ . Buyer utility as in Definition 2.2 is simply the difference between these two terms.


**2.4** **Marketplace Dynamics**

We can now formally define the per-step dynamic within the marketplace (see Figure 1 for a
graphical overview). Note this is a formalization of the steps laid out in the inventory optimization
example in Section 1.2. When Buyer _n_ arrives, the following steps occur in sequence (we assume
_p_ 0, _b_ 0, _Y_ 0 are initialized randomly):


For _n_ ∈[ _N_ ]:

(1) Market sets price _p_ _n_, where _p_ _n_ = PF ( _p_ _n_ −1, _b_ _n_ −1, _Y_ _n_ −1 )
(2) Buyer _n_ arrives with prediction task _Y_ _n_
(3) Buyer _n_ bids _b_ _n_ where _b_ _n_ = arg max _z_ ∈R + U( _z_, _Y_ _n_ )
(4) Market allocates features _X_ [�] _M_ to Buyer _n_, where _X_ [�] _M_ = AF ( _p_ _n_, _b_ _n_ ; _X_ _M_ )
(5) Buyer _n_ achieves G _Y_ _n_, M( _X_ [�] _M_ ) gain in prediction accuracy
� �
(6) Market extracts revenue, _r_ _n_, from Buyer _n_, where _r_ _n_ = RF ( _p_ _n_, _b_ _n_, _Y_ _n_ ; M, G)
(7) Market divides _r_ _n_ amongst allocated features using _ψ_ _n_, where _ψ_ _n_ = PD( _Y_ _n_, _X_ [�] _M_ ; M, G)


Remark 2.4. _A particularly important (albeit implicit) benefit of the above proposed architecture_
_is that the buyer’s do not ever access the underlying features. Rather they only receive predictions_


_through the ML model trained on the allocated features. This circumvents a known, difficult problem_
_in designing data markets where sellers are reluctant to release potentially valuable data streams as_
_they do not have control over who subsequently accesses it (since data streams are freely replicable)._


Remark 2.5. _In our proposed architecture, the price for each buyer is set centrally by the marketplace_
_rather than by the sellers individually. A sellers simply supplies data streams to the marketplace and is_
_assigned revenue based on the marginal contribution the data stream provides to the prediction task._
_Thus from the perspective of price setting, our model can equivalently be thought of as a single seller_
_supplying multiple data streams to the market and adjusting p_ _n_ _to maximize overall revenue._


Remark 2.6. _We note from the dynamics laid out above (specifically Step 3), a buyer is “myopic"_
_Y_ _over a single-stage i.e., Buyer_ ˆ _n_ _. Thus Buyer_ _n_ _maximizing utility only over Step_ _n_ _comes into the market once and leaves after being provided the estimate_ _n_ _. In particular, we do not study the additional_
_complication if the buyer’s utility is defined over multiple-stages._


Remark 2.7. _Our proposed architecture does not take into account an important attribute of data; a_
_firm’s utility for a particular dataset may be heavily dependent on what other firms get access to it_
_(e.g. a hedge fund might pay a premium to have a particularly predictive dataset only go to it). By_
_modeling buyer’s coming to the market one at a time, we do not study the externalities associated with_
_a dataset being replicated multiple times._


**3** **DESIRABLE PROPERTIES OF MARKETPLACE**

We define key properties for such a marketplace to robustly function in a large-scale, real-time
setting, where buyers are arriving in quick succession and need to be matched with a large number
of data sellers within minutes, if not quicker. Intuitively we require the following properties: (i)
buyers are truthful in their bids; (ii) overall revenue is maximized; (iii) revenue is fairly divided
amongst sellers; (iv) marketplace runs efficiently. In Sections 3.1-3.4, we formally define these
properties.


**3.1** **Truthfulness**


Property 3.1 (Truthful). _A marketplace is “truthful" if for all Y_ _n_ _,_


_µ_ _n_ = arg max U( _z_, _Y_ _n_ )
_z_ ∈R +


_where_ U( _z_, _Y_ _n_ ) _is defined as in Definition 2.2._


Property 3.1 requires that the allocation function, AF, and the revenue function, RF, incentivize
buyers to bid their true valuation for an increase in prediction accuracy. Note that we assume
buyers do not alter their prediction task, _Y_ _n_ .


**3.2** **Revenue Maximization**

Property 3.2 (Revenue Maximizing). _Let_ {( _µ_ 1, _b_ 1, _Y_ 1 ), ( _µ_ 2, _b_ 2, _Y_ 2 ), . . ., ( _µ_ _N_, _b_ _N_, _Y_ _N_ )} _be a se-_
_quence of buyers entering the market. A marketplace is “revenue maximizing" if the price-update_
_function,_ PF (·) _, produces a sequence of prices,_ { _p_ 1, _p_ 2, . . ., _p_ _n_ } _, such that the “worst-case" average_
_regret, relative to the optimal price p_ [∗] _in hindsight, goes to_ 0 _, i.e.,_



_N_
� RF ( _p_ _n_, _b_ _n_, _Y_ _n_ )�� .

_n_ =1



1
lim
_N_ →∞ _N_



sup
� {( _b_ _n_, _Y_ _n_ ): _n_ ∈[ _N_ ]}



sup
� _p_ [∗] ∈R +



_N_
� RF ( _p_ [∗], _b_ _n_, _Y_ _n_ ) −

_n_ =1



As is convention, we term the expression with the square bracket as regret and denote it, R( _N_, _M_ ) .
Property 3.2 is the standard worst-case regret guarantee (cf. [ 20 ]). It necessitates the price-update


function, PF, produce a sequence of prices _p_ _n_ such that the average difference with the unknown
optimal price in hindsight, _p_ [∗] goes to zero as _N_ increases. Note Property 3.2 must hold over the
worst case sequence of buyers i.e, no distributional assumptions on _µ_ _n_, _b_ _n_, _Y_ _n_ are made.


**3.3** **Revenue Division**


In the following section, we abuse notation and let _S_ ⊂[ _M_ ] refer to both the index of the training
features on sale and to the actual features, _X_ _S_ themselves.


_3.3.1_ _Shapley Fairness._


Property 3.3 (Shapley Fair). _A marketplace is “Shapley-fair" if_ ∀ _n_ ∈[ _N_ ], ∀ _Y_ _n_ _, the following_
_holds on_ PD _(and its output, ψ_ _n_ _):_

(1) _**Balance**_ _:_ [�] _m_ _[M]_ =1 _[ψ]_ _[n]_ [(] _[m]_ [)][ =][ 1]
(2) _**Symmetry**_ _:_ ∀ _m_, _m_ [′] ∈[ _M_ ], ∀ _S_ ⊂[ _M_ ] \ { _m_, _m_ [′] } _, if_ PD( _S_ ∪ _m_, _Y_ _n_ ) = PD( _S_ ∪ _m_ [′], _Y_ _n_ ) _, then_
_ψ_ _n_ ( _m_ ) = _ψ_ _n_ ( _m_ [′] )
(3) _**Zero Element**_ _:_ ∀ _m_ ∈[ _M_ ], ∀ _S_ ⊂[ _M_ ] _, if_ PD( _S_ ∪ _m_, _Y_ _n_ ) = PD( _S_, _Y_ _n_ ) _, then ψ_ _n_ ( _m_ ) = 0
(4) _**Additivity**_ _: Let the output of_ PD([ _M_ ], _Y_ _n_ [(][1][)] [)][,][ PD([] _[M]_ []][,] _[Y]_ _n_ [ (][2][)] [)] _[ be]_ _[ ψ]_ [ (] _n_ [1][)] [,] _[ψ]_ [ (] _n_ [2][)] _respectively. Let_ _ψ_ _n_ [′]
_be the output of_ PD([ _M_ ], _Y_ _n_ [(][1][)] + _Y_ _n_ [(][2][)] [)] _[. Then][ ψ]_ [ ′] _n_ [=] _[ ψ]_ [ (] _n_ [1][)] [+] _[ψ]_ [ (] _n_ [2][)] _[.]_


The conditions of Property 3.3, first laid out in [ 34 ], are considered the standard axioms of
fairness. We choose them as they are the de facto method to assess the marginal value of goods
(i.e., features in our setting) in a cooperative game (i.e., prediction task in our setting).


Remark 3.1. _A naive definition of the marginal value of feature_ _m_ _would be a “leave-one-out"_
_policy, i.e.,_ _ψ_ _n_ ( _m_ ) = G( _Y_ _n_, M( _X_ [�] [ _M_ ] )) −G( _Y_ _n_, M( _X_ [�] [ _M_ ]\ _m_ )) _. As the following toy example shows, the_
_correlation between features would lead to the market “undervaluing" each feature. Consider the simple_
_case where there are two sellers each selling identical features. It is easy to see the “leave-one-out"_
_policy above would lead to zero value being allocated to each feature, even though they collectively_
_might have great predictive value. This is clearly undesirable. That is why Property 3.3 is a necessary_
_notion of fairness as it takes into account the overlap of information that will invariably occur between_
_the different features, X_ _j_ _._


We then have the following celebrated theorem from [34],

Theorem 3.1 (Shapley Allocation). _Let_ _ψ_ _shapley_ ∈[ 0, 1 ] [[] _[M]_ []] _be the output of the following_
_algorithm,_



G _Y_ _n_, M( _X_ [�] _T_ ∪ _m_ ) −G _Y_ _n_, M( _X_ [�] _T_ ) (2)
� � � � � [�]



_ψ_ _shapley_ ( _m_ ) = �

_T_ ⊂[ _M_ ]\{ _m_ }



| _T_ |!( _M_ − | _T_ | − 1)!


_M_ !



_Then ψ_ _shapley_ _is the unique allocation that satisfies all conditions of Property 3.3_


Intuitively, this algorithm is computing the average marginal value of feature _m_ over all subsets
_T_ ⊂[ _M_ ] \ { _m_ } . It is easily seen that the running time of this algorithm is Θ( 2 _[M]_ ), which makes it
infeasible at scale if implemented as is. But it still serves as a useful standard to compare against.


_3.3.2_ _Robustness to Replication._

Property 3.4 (Robustness to replication). _For all_ _m_ ∈[ _M_ ] _, let_ _m_ _i_ [+] _[refer to the]_ _[ i]_ _[th]_ _[ replicated]_
_copy of_ _Let_ _ψ_ _n_ [+] _m_ [=] _i.e.,_ [ PD([] _X_ _m_ [+] _[M]_, _i_ []][=] [+] [,] _[ X][Y]_ _[n][m]_ [)] _[. Let]_ _[. Then a marketplace is]_ [ [] _[M]_ []] [+] [ =][ ∪] _[m]_ [(] _[m]_ [ ∪] _[i]_ _[ m][ ϵ]_ _i_ [+] [)] _[-“robust-to-replication" if]_ _[ refer to the set of original and replicated features.]_ [ ∀] _[n]_ [ ∈[] _[N]_ []][,][ ∀] _[Y]_ _[n]_ _[, the]_
_following holds on_ PD _:_
_ψ_ _n_ [+] [(] _[m]_ [)][ +] � _ψ_ _n_ [+] [(] _[m]_ _i_ [+] [) ≤] _[ψ]_ _[n]_ [(] _[m]_ [)][ +] _[ ϵ]_ [.]

_i_


Fig. 2. Shapley fairness is inadequate for freely replicable goods.


Property 3.4 is a novel notion of fairness, which can be considered a necessary additional
requirement to the Shapley notions of fairness for freely replicable goods. We use Example 3.1
below to elucidate how adverse replication of data can lead to grossly undesirable revenue divisions
(see Figure 2 for a graphical illustration). Note that implicit in the definition of Property 3.4 is that
the “strategy-space" of the data sellers is the number of times they replicate their data.


Example 3.1. _Consider a simple setting where the marketplace consists of only two sellers, A and B,_
_each selling one feature which are both identical. By Property 3.3, the Shapley value of A and B are_
_equal, i.e.,_ _ψ_ ( _A_ ) = [1] 2 [,] _[ψ]_ [(] _[B]_ [)][ =] [1] 2 _[. However if seller A replicated her feature once and sold it again in the]_

_marketplace, it is easy to see that the new Shapley allocation will be_ _ψ_ ( _A_ ) = [2] 3 [,] _[ψ]_ [(] _[B]_ [)][ =] [1] 3 _[. Hence it is]_

_not robust to replication since the aggregate payment remains the same (no change in accuracy)._


Such a notion of fairness in cooperative games is especially important in modern day applications
where: (i) digital goods are prevalent and can be produced at close to zero marginal cost; (ii) users get
utility from bundles of digital goods with potentially complex combinatorial interactions between
them. Two examples of such a setting are battery cost attribution among smartphone applications
and reward allocation among “experts" in a prediction market.


**3.4** **Computational Efficiency**


We assume the Machine Learning algorithm, M, and the Gain function, G, each require computation
running time of _O_ ( _M_ ), i.e., computation complexity scales at most linearly with the number of
features/sellers, _M_ . We define the following computational efficiency requirement of the market,


Property 3.5 (Efficient). _A marketplace is “efficient" if for each Step_ _n_ _, the marketplace as laid_
_out in Section 2.4 runs in polynomial time in_ _M_ _, where_ _M_ _is the number of sellers. In addition, the_
_computational complexity for each step of the marketplace cannot grow with N_ _._


Such a marketplace is feasible only if it functions in real-time. Thus, it is pertinent that the
computational resources required for any Buyer _n_ to interface with the market are low i.e., ideally
with run-time close to linear in _M_, the number of sellers, and not growing based on the number
of buyers seen thus far. Due to the combinatorial nature of data, this is a non–trivial requirement
as such combinatorial interactions normally lead to an exponential dependence in _M_ ; recall from
earlier sections, the Shapley Algorithm in Theorem 3.1 runs in Θ( 2 _[M]_ ) and a naive implementation
of Multiplicative Weights Algorithm for combinatorial goods runs in Θ(exp( _M_ )).


**4** **MARKETPLACE CONSTRUCTION**

We now explicitly construct instances of AF, RF, PF and PD and argue in Section 5 that the
properties laid out in Section 3 hold for these particular constructions.




[1] [1]

2 [,] _[ψ]_ [(] _[B]_ [)][ =] 2




[2] [1]

3 [,] _[ψ]_ [(] _[B]_ [)][ =] 3


Remark 4.1. _In line with Remark 2.5, we can think of_ AF, RF _as instances of how to design a_
_robust bidding, data allocation and revenue generation scheme from the buyer’s perspective with the_
_features sold held fixed (see Property 3.1). Analogously,_ PD _is a function for fair revenue division_
_from the seller’s perspective for a fixed amount of generated revenue (see Properties 3.3 and 3.4). And_
PF _is a function to centrally adjust the price of the features sold dynamically over time from the_
_marketplace’s perspective (see Property 3.2)._


**4.1** **Allocation and Revenue Functions (Buyer’s Perspective)**


**Allocation Function.** Recall the allocation function, AF, takes as input the current price _p_ _n_ and
the bid _b_ _n_ received, to decide the quality of the features _X_ _M_, used for Buyer _n_ ’s prediction task.
_Y_ ˆ _n_ received, rather than the particular datasets allocated. Thus the key structure we exploit inRecall from Definition 2.1 that a buyer’s utility comes solely from the quality of the estimate
designing AF is that from the buyer’s perspective, _instead of considering each feature_ _X_ _j_ _as a_
_separate good (which leads to computational intractability), it is greatly simplifying to think of_ _X_ _M_ _as_
_the total amount of “information" on sale_ . AF can thus be thought of as a function to _collectively_
adjust the quality of all of _X_ _M_ based on the difference between _p_ _n_ and _b_ _n_
_Specifically, we choose_ AF _to be a function that adds noise to/degrades_ _X_ _M_ _proportional to the_
_difference between_ _p_ _n_ _and_ _b_ _n_ . This degradation can take many forms and depends on the structure
of _X_ _j_ itself. Below, we provide examples of commonly used allocation functions for some typical
_X_ _j_ encountered in ML.


Example 4.1. _Consider_ _X_ _j_ ∈ R _[T]_ _i.e. sequence of real numbers. Then an allocation function (i.e._
_perturbation function),_ AF [∗] 1 [(] _[p]_ _[n]_ [,] _[b]_ _[n]_ [;] _[X]_ _[j]_ [)] _[, commonly used (cf. [13, 16]) would be for][ t][ in]_ [ [] _[T]_ []] _[,]_


˜
_X_ _j_ ( _t_ ) = _X_ _j_ ( _t_ ) + max(0, _p_ _n_ − _b_ _n_ ) · N(0, _σ_ [2] )


_where_ N(0, _σ_ [2] ) _is a univariate Gaussian._


Example 4.2. _Consider_ _X_ _j_ ∈{ 0, 1 } _[T]_ _i.e. sequence of bits. Then an allocation function (i.e. masking_
_function),_ AF [∗] 2 [(] _[p]_ _[n]_ [,] _[b]_ _[n]_ [;] _[X]_ _[j]_ [)] _[, commonly used (cf. [33]) would be for][ t][ in]_ [ [] _[T]_ []] _[,]_


˜
_X_ _j_ ( _t_ ) = _B_ ( _t_ ; _θ_ ) · _X_ _j_ ( _t_ )

_where B_ ( _t_ ; _θ_ ) _is an independent Bernoulli random variable with parameter θ_ = min( _p_ _[b]_ _[n]_ _n_ [,][ 1][)] _[.]_


In both examples if _b_ _n_ ≥ _p_ _n_, then the buyer is given _X_ _M_ as is without degradation. However if
_b_ _n_ < _p_ _n_, then _X_ _j_ is degraded in proportion to the difference between _b_ _n_ and _p_ _n_ .


Remark 4.2. _Through Assumption 1 (see Section 5.1), we formalize a natural and necessary property_
_required of any such allocation function so that Property 3.1 (truthfulness) holds. Specifically, for a_
_fixed price_ _p_ _n_ _, increasing the bid_ _b_ _n_ _cannot lead to a decrease in prediction quality. The space of possible_
_allocations functions that meet this criteria is clearly quite large. We leave it as future work to study_
_what is the optimal_ AF _from the space of feasible allocation functions to maximize revenue._


Remark 4.3. _A celebrated result from [_ _29_ _] is that for single-parameter buyers, a single take-it-or-_
_leave-it price for all data is optimal, i.e. if the bid is above the singe posted price then allocate all_
_the data without any noise and if the bid is less than the price, allocate no data. However, maybe_
_surprisingly, in our setting this result does not apply. This is due to an important subtlety in our_
_formalism - while_ _µ_ _n_ _(how much a buyer values a marginal increase in accuracy), is a scalar, a buyer_
_is also parametrized by Y_ _n_ _, the prediction task._
_This leads to the following simple counter-example - imagine buyers are only of two types: (i) Type I_
_with prediction task_ _Y_ 1 _and valuation_ _µ_ 1 _; (ii) Type II with prediction task_ _Y_ 2 _and valuation_ _µ_ 2 _. Further,_


Fig. 3. Features allocated (AF [∗] ) and revenue collected (RF [∗] ) for a particular price vector _p_ _n_ and bid _b_ _n_ .


_let there be only two types of features on sale,_ _X_ 1 _and_ _X_ 2 _. Assume_ _X_ 1 _is perfectly predictive of prediction_
_task_ _Y_ 1 _and has zero predictive value for_ _Y_ 2 _. Analogously, assume_ _X_ 2 _is perfectly predictive of_ _Y_ 2 _and_
_has no predictive value for_ _Y_ 1 _. Then it is easy to see that the optimal pricing mechanism is to set the_
_price of X_ 1 _to be µ_ 1 _and X_ 2 _to be µ_ 2 _. Thus a single posted price is not optimal_ [6] _._
_More generally, if different datasets have varying amounts of predictive power for different buyer_
_types, it is not even clear that a take-it-or-leave-it price per feature sold is optimal._


**Revenue Function.** Recall from Definition 2.1, we parameterize buyer utility through the parameter _µ_ _n_, i.e., how much a buyer values a marginal increase in prediction quality. This crucial
modeling choice allows us to use Myerson’s payment function rule (cf. [29]) given below,


_b_ _n_
RF [∗] ( _p_ _n_, _b_ _n_, _Y_ _n_ ) = _b_ _n_   - G _Y_ _n_, M AF [∗] ( _b_ _n_, _p_ _n_ ) − G _Y_ _n_, M AF [∗] ( _z_, _p_ _n_ ) _dz_ . (3)
� � � [�] ∫ 0 � � � [�]


In Theorem 5.1, we show that RF [∗] _ensures Buyer_ _n_ _is truthful_ (as defined in Property 3.1). Refer to
Figure 3 for a graphical view of AF [∗] and RF [∗] .


**4.2** **Price Update Function (Marketplace Perspective)**

The market is tasked with how to pick _p_ _n_ for _n_ ∈[ _N_ ] . Recall from Section 2.4, the market must
decide _p_ _n_ before Buyer _n_ arrives (otherwise, it is easily seen that truthfulness cannot be gauranteed).
We now provide some intuition of how increasing/decreasing _p_ _n_ affects the amount of revenue
collected, and the implicit tradeoff that lies therein. Observe from the construction of RF [∗] in (3)
that for a fixed bid, _b_ _n_, and prediction task _Y_ _n_, it is easily seen that if _p_ _n_ is picked to be too large,
then the positive term in RF [∗] is small (as the degradation of the signal in _X_ _M_ is very high), leading
to lower than optimal revenue generation. Similarly, if _p_ _n_ is picked to be too small, it is easily seen
that the negative term in RF [∗] is large, which again leads to an undesired loss in revenue.
In Algorithm 1, we apply the Multiplicative Weights method to pick _p_ _n_ in an online fashion
and to balance the tradeoff described above. To construct Algorithm 1 more precisely, we need to
define some additional quantities. As we make precise in Assumption 4 in Section 5, we assume
bids come from some bounded set, B ⊂ R + . We define B max ∈ R to be the maximum element of B


6 When it comes to truthfulness (see Property 3.1), the fact that we can parametrize a buyerâĂŹs valuation through a
scalar, _µ_ _n_, does indeed mean MyersonâĂŹs payment function (see (3) ) is truthful as long as the data allocation function is

monotonic.


**ALGORITHM 1:** PRICE-UPDATE: PF [∗] ( _b_ _n_, _Y_ _n_, B, _ϵ_, _δ_ )

**Input:** _b_ _n_, _Y_ _n_, B, _ϵ_, _δ_
**Output:** _p_ _n_
Let B net ( _ϵ_ ) be an _ϵ_ -net of B;
**for** _c_ _[i]_ ∈B _net_ ( _ϵ_ ) **do**
Set _w_ 1 _[i]_ [=][ 1 ;] // initialize weights of all experts to 1
**end**

**for** _n_ = 1 _to N_ **do**
_W_ _n_ = [�] _i_ [|B] =1 [net] [(] _[ϵ]_ [)|] _w_ _n_ _[i]_ ;
Let _p_ _n_ = _c_ _[i]_ with probability _w_ _n_ _[i]_ / _W_ _n_ ; // note _p_ _n_ is not a function of _b_ _n_
**for** _c_ _[i]_ ∈B _net_ ( _ϵ_ ) **do**
Let _д_ _n_ _[i]_ = RF [∗] ( _c_ _[i]_, _b_ _n_, _Y_ _n_ )/B max ; // revenue gain if price _c_ _[i]_ was used
Set _w_ _n_ _[i]_ +1 [=] _[ w]_ _[in]_ [ · (][1][ +] _[ δд]_ _[in]_ [)][ ;] // Multiplicative Weights update step
**end**

**end**


and B net ( _ϵ_ ) to be a minimal _ϵ_ -net of B [7] . Intuitively, the elements of B net ( _ϵ_ ) serve as our “experts"
(i.e. the different prices we experiment with) in the Multiplicative Weights algorithm.
In Theorem 5.2, we show that this algorithm does indeed achieve zero-regert with respect to the
optimal _p_ [∗] ∈ R + in hindsight.


**4.3** **Payment-Division Functions (Seller’s Perspective)**

_4.3.1_ _Shapley Approximation._ In our model (see Section 2.4), a buyer only makes an aggregate
payment to the market based on the increase in accuracy experienced (see RF [∗] in (3) ). It is thus
up to the market to design a mechanism to fairly (as defined in Property 3.3) allocate the revenue
among the sellers to incentivize their participation. Following the seminal work in [ 34 ], there
have been a substantial number of applications (cf. [ 6, 7 ]) leveraging the ideas in [ 34 ] to fairly
allocate cost/reward among strategic entities cooperating towards a common goal. Since the Shapley
algorithm stated in (2) is the unique method to satisfy Property 3.3, but unfortunately runs in time
Θ(2 _[M]_ ), the best one can do is to approximate (2) as closely as possible.
In Algorithm 2, we uniformly sample from the space of permutations over [ _M_ ] to construct an
approximation of the Shapley value in (2) . To construct Algorithm 2 more precisely, we need to
define some additional quantities. Let _σ_ [ _M_ ] refer to the set of all permutations over [ _M_ ] . For any
permutation _σ_ ∈ _σ_ [ _M_ ], let [ _σ_ < _m_ ] refer to the set of features in [ _M_ ] that came before _m_ .
The key observation in showing that Algorithm 2 is effective, is that instead of enumerating
over all permutations in _σ_ [ _M_ ] as in the Shapley allocation, it suffices to sample _σ_ _k_ ∈ _σ_ [ _M_ ] uniformly
at random with replacement, _K_ times, where _K_ depends on the _ϵ_ -approximation a practitioner
desires. We provide guidance on how to pick _K_ in Section 5.4. We note some similar sampling based
methods, albeit for different applications (cf. [11, 25, 26]).
In Theorem 5.3, we show that Algorithm 2 gives an _ϵ_ -approximation for (2) with high probability
while running in time _O_ ( _M_ [2] ).


_4.3.2_ _Robustness to Replication._ Recall from Section 3.3.2 that for freely replicable goods such as
data, the standard Shapley notion of fairness does not suffice (see Example 3.1 for how it can lead
to undesirable revenue allocations). Though this issue may seem difficult to overcome in general,


7 We endow R with the standard Euclidean metric. An _ϵ_ -net of a set B is a set _K_ ⊂B such that for every point _x_ ∈B,
there is a point _x_ 0 ∈ _K_ such that | _x_ − _x_ 0 | ≤ _ϵ_ .


**ALGORITHM 2:** SHAPLEY-APPROX: PD [∗]
_A_ [(] _[Y]_ _[n]_ [,][ �] _[X]_ _[ M]_ [,] _[K]_ [)]



**Input:** _Y_ _n_, _X_ [�] _M_, _K_
**Output:** _ψ_ [ˆ] _n_ = [ _ψ_ [ˆ] _n_ ( _m_ ) : _m_ ∈[ _M_ ]]
Let B net ( _ϵ_ ) be an _ϵ_ -net of B;
**for** _m_ ∈[ _M_ ] **do**
**for** _k_ ∈[ _K_ ] **do**
_σ_ _k_ ∼ Unif( _σ_ [ _M_ ] );
_G_ = G( _Y_ _n_, M( _X_ [ _σ_ _k_ < _m_ ] ));
_G_ ˆ [+] = G( _Y_ _n_, M( _X_ [ _σ_ _k_ < _m_ ∪ _m_ ] ));
_ψ_ _n_ _[k]_ ( _m_ ) = [ _G_ [+] − _G_ ]
**end**

ˆ _K_
_ψ_ _n_ ( _m_ ) = _K_ [1] � _k_ =1 _[ψ]_ [ˆ] _[kn]_ [ (] _[m]_ [)]

**end**


we again exploit the particular structure of data as a path forward. Specifically, we note that there
are _standard methods to define the “similarity" between two vectors of data_ . A complete treatment of
similarity measures has been done in [18]. We provide two examples:


Example 4.3. _Cosine similarity, a standard metric used in text mining and information retrieval, is:_

|⟨ _X_ 1, _X_ 2 ⟩|
, _X_ 1, _X_ 2 ∈ R _[T]_
|| _X_ 1 || 2 || _X_ 2 || 2


Example 4.4. _“Inverse" Hellinger distance, a standard metric to define similarity between underlying_
_data distributions, is:_ 1 − [1] 2 � _x_ ∈X [(] � _p_ 1 ( _x_ ) − � _p_ 2 ( _x_ )) [2] ) [1][/][2], _p_ 1 ∼ _X_ 1, _p_ 2 ∼ _X_ 2 _._


We introduce some natural properties any such similarity metric must satisfy for our purposes,


Definition 4.1 ( **Adapted from [18]** ). _A similarity metric is a function,_ SM : R _[T]_ ×R _[T]_ →[ 0, 1 ] _,_
_that satisfies: (i) Limited Range:_ 0 ≤SM(·, ·) ≤ 1 _; (ii) Reflexivity:_ SM( _X_, _Y_ ) = 1 _if and only if_ _X_ = _Y_ _;_
_(iii) Symmetry:_ SM( _X_, _Y_ ) = SM( _Y_, _X_ ) _; (iv) Define_ _d_ SM( _X_, _Y_ ) = 1 −SM( _X_, _Y_ ) _, then Triangle_
_Inequality: d_ SM( _X_, _Y_ ) + _d_ SM( _Y_, _Z_ ) ≥ _d_ SM( _X_, _Z_ )


In Algorithm 3, we construct a “robust-to-replication" version of the randomized Shapley approximation algorithm by utilizing Definition 4.1 above.
Intuitively, the algorithm penalizes similar features (relative to the similarity metric, SM ) to
disincentivize replication. We provide guidance on how to pick the hyper-parameter _λ_ in Section 5.
In Theorem 5.4, we show Algorithm 3 is _ϵ_ -“Robust to Replication" i.e. Property 3.4 (Robustnessto-Replication) holds. See the example below for an illustration of the effect of Algorithm 3 on
undesired replication.


Example 4.5. _Recall Example 3.1 where there are two sellers, A and B, each selling an identical_
_feature. In that example, if Seller A replicated her feature, her Shapley allocation increased from_ [1] 2

_to_ [2]

3 _[. If we instead apply Algorithm 3 (with]_ _[ λ]_ [ =] [ 1] _[), then it is easy to see that her Shapley allocation]_
_decreases from_ [1] _[to]_ 2 [2] _[, ensuring Property 3.4 holds. See Figure 4 for an illustration.]_




[1] 2 � _x_ ∈X [(] �



_p_ 1 ( _x_ ) − �



_p_ 2 ( _x_ )) [2] ) [1][/][2], _p_ 1 ∼ _X_ 1, _p_ 2 ∼ _X_ 2 _._



2 [1] _e_ _[to]_ 3 _e_ 2 [2] _[, ensuring Property 3.4 holds. See Figure 4 for an illustration.]_



**5** **MAIN RESULTS**


**5.1** **Assumptions.**

To give performance guarantees, we state four mild and natural assumptions we need on: (i) AF [∗]

(allocation function); (ii) M (ML algorithm); (iii) RF [∗] (revenue function); (iv) _b_ _n_ (bids made).


**ALGORITHM 3:** SHAPLEY-ROBUST: PD [∗]
_B_ [(] _[Y]_ _[n]_ [,][ �] _[X]_ _[ M]_ [,] _[K]_ [,][ SM][,] _[ λ]_ [)]

**Input:** _Y_ _n_, _X_ [�] _M_, _K_, SM, _λ_
_ψ_ **Output:** ˆ _n_ ( _m_ ) = SHAPLEY-APPROX( _ψ_ _n_ = [ _ψ_ _n_ ( _m_ ) : _m_ ∈[ _MY_ _n_ ]], M, G, _K_ );
_ψ_ _n_ ( _m_ ) = _ψ_ [ˆ] _n_ ( _m_ ) exp(− _λ_ [�] _j_ ∈[ _M_ ]\{ _m_ } [SM(] _[X]_ _[m]_ [,] _[X]_ _[j]_ [))][;]


Fig. 4. A simple example illustrating how SHAPLEY-ROBUST down weights similar data to ensure robustness
to replication.


Assumption 1 ( AF [∗] is Monotonic). M, AF [∗] _are such that an increase in the difference between_
_p_ _n_ _and_ _b_ _n_ _leads to a decrease in_ G _i.e. an increase in “noise" cannot lead to an increase in prediction accu-_

_racy. Specifically, for any_ _Y_ _n_, _p_ _n_ _, let_ _X_ [�] ( _M_ 1) [,][ �] _[X]_ ( _M_ 2) _[be the outputs of]_ [ AF (] _[p]_ _[n]_ [,] _[b]_ [(][1][)] [;] _[X]_ _[ M]_ [)][,][ AF (] _[p]_ _[n]_ [,] _[b]_ [(][2][)] [;] _[X]_ _[ M]_ [)]

(1) (2)
_respctively. Then if b_ [(][1][)] ≤ _b_ [(][2][)] _, we have_ G _Y_ _n_, M( _X_ [�] _M_ [)] ≤G _Y_ _n_, M( _X_ [�] _M_ [)] _._
� � � �


Assumption 2 ( M is Invariant to Replicated Data). M _is such that replicated features do not_
_cause a change in prediction accuracy. Specifically,_ ∀ _S_ ⊂[ _M_ ], ∀ _Y_ _n_, ∀ _m_ ∈ _S_ _, let_ _m_ _i_ [+] _[refer to the]_ _[ i]_ _[th]_

_replicated copy of_ _m_ _(i.e._ _X_ _m_ [+], _i_ [=] _[ X]_ _[m]_ _[). Let]_ _[ S]_ [+] [ =][ ∪] _[m]_ [(] _[m]_ [ ∪] _[i]_ _[ m]_ _i_ [+] [)] _[ refer to the set of original and replicated]_
_features. Then_ G( _Y_ _n_, M( _X_ _S_ )) = G( _Y_ _n_, M( _X_ _S_ [+] ))


Assumption 3 ( RF [∗] is Lipschitz). _The revenue function_ RF [∗] _is_ L _-Lipschitz with respect to price._
_Specifically, for any_ _Y_ _n_, _b_ _n_, _p_ [(][1][)], _p_ [(][2][)] _, we have_ |RF [∗] ( _p_ [(][1][)], _b_ _n_, _Y_ _n_ ) −RF [∗] ( _p_ [(][2][)], _b_ _n_, _Y_ _n_ )| ≤L| _p_ [(][1][)] − _p_ [(][2][)] | _._


Assumption 4 (Bounded Bids). _The set of possible bids_ _b_ _n_ _for_ _n_ ∈[ _N_ ] _come from a closed, bounded_
_set_ B _. Specifically, b_ _n_ ∈B _, where diameter_ (B) = _D, where D_ < ∞ _._


Remark 5.1. _We provide some justification for Assumptions 1 and 2 above, which impose require-_
_ments of the ML algorithm and the accuracy metric (i.e._ M _and_ G _). These assumptions require that:_
_(i) as more noise is added to data, the less gain in prediction accuracy; (ii) replicated features do not_
_have an effect on the accuracy. Essentially all ML algorithms and the accuracy metrics function like_
_this. Thus these assumptions reflects standard, weak statistical assumptions. Assumptions 3 and 4 are_
_self-explanatory._


**5.2** **Truthfulness.**

Theorem 5.1. _For_ AF [∗] _, Property 3.1 (Truthfulness) can be achieved if and only if Assumption 1_
_holds. In which case,_ RF [∗] _guarantees truthfulness._


Theorem 5 . 1 is an application of Myerson’s payment function (cf. [ 29 ]) which ensures _b_ _n_ = _µ_ _n_ .
See Appendix A for the proof.


Again, the key is the modeling choice made to define buyer utility as in Definition 2.1. It lets
us parameterize a buyers value for increased accuracy by a scalar, _µ_ _n_, which allows us to exploit
Myerson’s payment function (unfortunately generalization of Myerson’s payment function to the
setting where _µ_ _n_ is a vector are severely limited cf. [14]).


**5.3** **Revenue Maximization.**


Theorem 5.2. _Let Assumptions 1, 3 and 4 hold. Let_ _p_ _n_ : _n_ ∈[ _N_ ] _be the output of Algorithm 1. Let_ L _be_
_the Lipschitz constant of_ RF [∗] _(defined as in Assumption 3). Let_ B max ∈ R _be the maximum element_
_of_ B _(where_ B _is defined as in Assumption 4). Then by choosing the algorithm hyper-parameters_
_ϵ_ = (L√ _N_ ) [−][1], _δ_ = �log(|B _net_ ( _ϵ_ )|)/ _N_ _, the total average regret is bounded by,_



_N_ ) [−][1], _δ_ = �



log(|B _net_ ( _ϵ_ )|)/ _N_ _, the total average regret is bounded by,_



~~�~~



log( _N_ )

),
_N_



1
_N_ [E][[R(] _[N]_ [)] ≤] _[C]_ [B] [max]



~~�~~



log(B max L ~~√~~ _N_ )

= _O_ (
_N_



_for some positive constant_ _C_ - 0 _. Here, the expectation is taken over the randomness in Algorithm 1._
_Hence, Property 3.2 (Revenue Maxmization) holds._


Theorem 5.2 proves Algorithm 1 is a zero regret algorithm. We note the bound is independent of
_M_, the number of features sold. See Appendix B for the proof.
As we note in Remark 4.2, a limitation of the AF [∗] we design is that it is fixed, i.e., we degrade
each feature by the same scaling. We leave it as future work to design an adaptive AF ; instead
of fixing AF [∗] a priori (as we currently do using standard noising procedures), can we make
the noising procedure adaptive to the prediction tasks to further increase the revenue generated
(potentially by adding distributional assumptions to the prediction tasks)?


**5.4** **Fairness in Revenue Division.**


Theorem 5.3. _Let_ _ψ_ _n_, _shapley_ _be the unique vector satisfying Property 3.3 (Shapley Fairness) as given_
_in_ (2) _. For Algorithm 2, pick the following hyperparameter:_ _K_ - ( _M_ log( 2 / _δ_ ))/( 2 _ϵ_ [2] ) _, where_ _δ_, _ϵ_ - 0 _._
_Then with probability_ 1 − _δ_ _, the output_ _ψ_ [ˆ] _n_ _of Algorithm 2, achieves the following,_

|| _ψ_ _n_, _shapley_ − _ψ_ [ˆ] _n_ || ∞ < _ϵ_ .


Theorem 5.3 gives an _ϵ_ -approximation for _ψ_ _n_,shapley, the _unique_ vector satisfying Property 3.3, in
_O_ ( _M_ ) . Recall, computing it exactly would take Θ( 2 _[M]_ ) running time. See Appendix C for the proof.
To the best of our knowledge, the direct application of random sampling to compute feature
importances for ML algorithms along with finite sample guarantees is novel. We believe this random
sampling method could be used as a model-agnostic tool (not dependent on the particulars of
the prediction model used) to assess feature importance - a prevalent question for data scientists
seeking interpretability from their prediction models.


Theorem 5.4. _Let Assumption 2 hold. For Algorithm 3, pick the following hyperparameters:_ _K_ ≥
( _M_ log( 2 / _δ_ ))/( 2 ( _ϵ_ / 3 ) [2] ), _λ_ = log( 2 ) _, where_ _δ_, _ϵ_ - 0 _. Then with probability_ 1 − _δ_ _, the output,_ _ψ_ _n_ _, of_
_Algorithm 3 is_ _ϵ_ _-“Robust to Replication" i.e. Property 3.4 (Robustness-to-Replication) holds. Additionally_
_Conditions 2-4 of Property 3.3 continue to hold for ψ_ _n_ _with ϵ-precision._


Theorem 5.4 states Algorithm 3 protects against adversarial replication of data, while maintaining
the conditions of the standard Shapley fairness other than balance. Again, the key observation,
which makes Algorithm 3 possible is that we can precisely compute similarity between data streams
(see Definition 4.1). See Appendix C for the proof.
A natural question is whether Property 3.4 and Condition 1 of Property 3.3 and can hold together.
Unfortunately, as we see from Proposition 5.1, they cannot (see Appendix C for the proof),


Proposition 5.1. _If the identities of sellers in the marketplace is anonymized, the balance condition_
_in Property 3.3 and Property 3.4 cannot simultaneously hold._


Note however, Algorithm 3, down-weights features in a “local" fashion i.e. highly correlated
features are individually down-weighted, while uncorrelated features are not. _Hence, Algorithm 3_
_incentivizes sellers to provide data that is: (i) predictive for a wide variety of tasks; (ii) uncorrelated_
_with other features on sale i.e. has unique information_ .
In Step 2 of Algorithm 3, we exponentially penalize (i.e. down weight) each feature, for a given
similarity metric, SM . An open question for future work is which revenue division mechanism
is the most balanced preserving while being robust to replication? As an important first step, we
provide a necessary and sufficient condition for any penalty function [8] to be robust to replication
for a given similarity metric, SM (see Appendix E for the proof),


Proposition 5.2. _Let Assumption 2 hold. Then for a given similarity metric_ SM _, a penalty function_
_f is “robust-to-replication" if and only if it satisfies the following relation for any c_ ∈ Z +, _x_ ∈ R + _,_


( _c_ + 1) _f_ ( _x_ + _c_ ) ≤ _f_ ( _x_ )


**5.5** **Efficiency.**

Corollary 5.1. AF [∗], RF [∗], PF [∗] _run in O_ ( _M_ ) _._ PD _a_ [∗] _[,]_ [ PD] _b_ [∗] _[run in][ O]_ [(] _[M]_ [2] [)] _[. Property 3.5 holds.]_

See Appendix D for the proof. AF [∗], RF [∗], PF [∗] running in _O_ ( _M_ ) is desirable as they need to be
re-computed in real-time for every buyer. However, the revenue division algorithms (which run in
_O_ ( _M_ [2] )) can conceivably run offline as we assume the sellers to be fixed.


**6** **CONCLUSION**


**Modeling contributions.** Our main contribution is a mathematical model for a two-sided data
market (Section 2). We hope our proposed architecture can serve as a foundation to operationalize
real-time data marketplaces, which have applicability in a wide variety of important commercial
settings (Section 1.2). To further this goal, we define key challenges (Section 3), construct algorithms
to meet these challenges (Section 4) and theoretically analyze their performance (Section 5).
To make the problem tractable, we make some key modeling choices. Two of the most pertinent
ones include: (i) Buyer _n_ ’s utility comes solely from the quality of the estimate _Y_ [ˆ] _n_ received, rather
than the particular datasets allocated (Definition 2.1); (ii) the marketplace is allowed to centrally set
prices for all features for each buyer, rather than sellers individually setting prices for each feature
(Remark 2.5).


**Technical contributions.** We highlight two technical contributions, which might be of independent interest. First, a new notion of “fairness" required for cooperative games with freely replicable
goods (and associated algorithms). As stated earlier (Section 3.3.2), such a notion of fairness is especially important in modern applications where users get utility/cost from bundles of digital goods
with potentially complex combinatorial interactions (e.g. battery cost attribution for smartphone
applications, reward allocation in prediction markets). Second, a truthful, zero regret mechanism
for auctioning a particular class of combinatorial goods, which utilizes Myerson’s payment function
and the Multiplicative Weights algorithm. Specifically, if one can find a way of modeling buyer
utility/cost through a scalar parameter (e.g. number of unique views for multimedia ad campaigns,
total battery usage for smartphone apps), then the framework described can potentially be applied.


8 We define a general penalty function to be of the form ˆ _ψ_ _n_ ( _m_ ) _f_ (·), instead of ˆ _ψ_ _n_ ( _m_ ) exp(− _λ_ [�] _j_ ∈[ _M_ ]\{ _m_ } [SM(] _[X]_ _m_ [,] _[ X]_ _j_ [))]
as in Step 2 of Algorithm 3.


**Future Work.** We reiterate some interesting lines of questioning for future work: (i) how to take
into account the externalities of replication experienced by buyers (Remark 2.7); (ii) how to design
an adaptive allocation function that further increases revenue generated (Remark 4.2); (iii) which
“robust-to-replication" revenue division mechanism is the most balanced preserving (Section 5.4)?


**ACKNOWLEDGMENTS**

During this work, the authors were supported in part by a MIT Institute for Data, Systems and
Society (IDSS) WorldQuant and Thompson Reuters Fellowship.


**REFERENCES**


[1] Martín Abadi, Paul Barham, Jianmin Chen, Zhifeng Chen, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat,
Geoffrey Irving, Michael Isard, Manjunath Kudlur, Josh Levenberg, Rajat Monga, Sherry Moore, Derek G. Murray,
Benoit Steiner, Paul Tucker, Vijay Vasudevan, Pete Warden, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng. 2016.
TensorFlow: A System for Large-scale Machine Learning. In _Proceedings of the 12th USENIX Conference on Operating_
_Systems Design and Implementation (OSDI’16)_ [. USENIX Association, Berkeley, CA, USA, 265–283. http://dl.acm.org/](http://dl.acm.org/citation.cfm?id=3026877.3026899)
[citation.cfm?id=3026877.3026899](http://dl.acm.org/citation.cfm?id=3026877.3026899)

[2] Bill Aiello, Yuval Ishai, and Omer Reingold. 2001. Priced Oblivious Transfer: How to Sell Digital Goods. In _International_
_Conference on the Theory and Applications of Cryptographic Techniques_ . Springer, 119–135.

[3] Sanjeev Arora, Elad Hazan, and Satyen Kale. 2012. The Multiplicative Weights Update Method: a Meta-Algorithm and
Applications. _Theory of Computing_ [8, 6 (2012), 121–164. https://doi.org/10.4086/toc.2012.v008a006](https://doi.org/10.4086/toc.2012.v008a006)

[4] Peter Auer. 2003. Using Confidence Bounds for Exploitation-exploration Trade-offs. _Journal of Machine Learning_
_Research_ [3 (March 2003), 397–422. http://dl.acm.org/citation.cfm?id=944919.944941](http://dl.acm.org/citation.cfm?id=944919.944941)

[5] Moshe Babaioff, Robert Kleinberg, and Renato Paes Leme. 2012. Optimal Mechanisms for Selling Information. In
_Proceedings of the 13th ACM Conference on Electronic Commerce (EC ’12)_ . ACM, New York, NY, USA, 92–109.

[6] Yoram Bachrach, Evangelos Markakis, Ezra Resnick, Ariel D. Procaccia, Jeffrey S. Rosenschein, and Amin Saberi. 2010.
Approximating power indices: theoretical and empirical analysis. _Autonomous Agents and Multi-Agent Systems_ 20, 2
(01 Mar 2010), 105–122.

[7] Eric Balkanski, Umar Syed, and Sergei Vassilvitskii. 2017. Statistical Cost Sharing. In _Advances in Neural Information_
_Processing Systems 30_ [. Curran Associates, Inc., 6221–6230. http://papers.nips.cc/paper/7202-statistical-cost-sharing.pdf](http://papers.nips.cc/paper/7202-statistical-cost-sharing.pdf)

[8] Siddhartha Banerjee, Ramesh Johari, and Carlos Riquelme. 2015. Pricing in Ride-Sharing Platforms: A QueueingTheoretic Approach. In _Proceedings of the Sixteenth ACM Conference on Economics and Computation (EC ’15)_ . ACM,
[New York, NY, USA, 639–639. https://doi.org/10.1145/2764468.2764527](https://doi.org/10.1145/2764468.2764527)

[9] Dirk Bergemann, Alessandro Bonatti, and Alex Smolin. 2018. The Design and Price of Information. _American Economic_
_Review_ [108, 1 (January 2018), 1–48. https://doi.org/10.1257/aer.20161079](https://doi.org/10.1257/aer.20161079)

[10] Bernard Caillaud and Bruno Jullien. 2003. Chicken & Egg: Competition Among Intermediation Service Providers. _The_
_RAND Journal of Economics_ [34, 2 (2003), 309–328. http://www.jstor.org/stable/1593720](http://www.jstor.org/stable/1593720)

[11] Javier Castro, Daniel Gómez, and Juan Tejada. 2009. Polynomial Calculation of the Shapley Value Based on Sampling.
_Computer and Operations Research_ [36, 5 (May 2009), 1726–1730. https://doi.org/10.1016/j.cor.2008.04.004](https://doi.org/10.1016/j.cor.2008.04.004)

[12] M. Keith Chen. 2016. Dynamic Pricing in a Labor Market: Surge Pricing and Flexible Work on the Uber Platform. In
_Proceedings of the 2016 ACM Conference on Economics and Computation (EC ’16)_ . ACM, New York, NY, USA, 455–455.
[https://doi.org/10.1145/2940716.2940798](https://doi.org/10.1145/2940716.2940798)

[13] Rachel Cummings, Katrina Ligett, Aaron Roth, Zhiwei Steven Wu, and Juba Ziani. 2015. Accuracy for Sale: Aggregating
Data with a Variance Constraint. In _Proceedings of the 2015 Conference on Innovations in Theoretical Computer Science_
_(ITCS ’15)_ [. ACM, New York, NY, USA, 317–324. https://doi.org/10.1145/2688073.2688106](https://doi.org/10.1145/2688073.2688106)

[14] Constantinos Daskalakis. 2015. Multi-item Auctions Defying Intuition? _SIGecom Exch._ 14, 1 (Nov. 2015), 41–75.
[https://doi.org/10.1145/2845926.2845928](https://doi.org/10.1145/2845926.2845928)

[15] C. Daskalakis and V. Syrgkanis. 2016. Learning in Auctions: Regret is Hard, Envy is Easy. In _2016 IEEE 57th Annual_
_Symposium on Foundations of Computer Science (FOCS)_ [. 219–228. https://doi.org/10.1109/FOCS.2016.31](https://doi.org/10.1109/FOCS.2016.31)

[16] Arpita Ghosh and Aaron Roth. 2011. Selling Privacy at Auction. In _Proceedings of the 12th ACM Conference on Electronic_
_Commerce (EC ’11)_ [. ACM, New York, NY, USA, 199–208. https://doi.org/10.1145/1993574.1993605](https://doi.org/10.1145/1993574.1993605)

[17] Renato Gomes. 2014. Optimal Auction Design in Two-Sided Markets. _The RAND Journal of Economics_ 45, 2 (2014),

248–272.

[18] A Ardeshir Goshtasby. 2012. Similarity and Dissimilarity Measures. In _Image Registration_ . Springer, 7–66.

[19] Robin Hanson. 2012. Logarithmic markets coring rules for modular combinatorial information aggregation. _The_
_Journal of Prediction Markets_ 1, 1 (2012), 3–15.


[20] Elad Hazan et al . 2016. Introduction to online convex optimization. _Foundations and Trends_ _®_ _in Optimization_ 2, 3-4
(2016), 157–325.

[21] Daniel P Heyman and Matthew J Sobel. 2004. _Stochastic Models in Operations Research, Volume II. Stochastic Optimization_ .
Vol. 2. Courier Corporation.

[22] Katrina Ligett and Aaron Roth. 2012. Take It or Leave It: Running a Survey When Privacy Comes at a Cost. In _Internet_
_and Network Economics_, Paul W. Goldberg (Ed.). Springer Berlin Heidelberg, Berlin, Heidelberg, 378–391.

[23] De Liu and Jianqing Chen. 2006. Designing online auctions with past performance information. _Decision Support_
_Systems_ 42, 3 (2006), 1307–1320.

[24] Yungao Ma, Nengmin Wang, Ada Che, Yufei Huang, and Jinpeng Xu. 2013. The Bullwhip Effect on Product Orders and
Inventory: A Perspective of Demand Forecasting Techniques. _International Journal of Production Research_ 51, 1 (2013),

281–302.

[25] Sasan Maleki, Long Tran-Thanh, Greg Hines, Talal Rahwan, and Alex Rogers. 2013. Bounding the Estimation Error of
Sampling-based Shapley Value Approximation With/Without Stratifying. _CoRR_ abs/1306.4265 (2013).

[26] I Mann and LS Shapley. 1952. _Values for large games IV: Evaluating the electoral college exactly_ . Technical Report.
RAND Corp Santa Monica CA.

[27] Aranyak Mehta et al . 2013. Online matching and ad allocation. _Foundations and Trends_ _®_ _in Theoretical Computer_
_Science_ 8, 4 (2013), 265–368.

[28] Xiangrui Meng, Joseph Bradley, Burak Yavuz, Evan Sparks, Shivaram Venkataraman, Davies Liu, Jeremy Freeman, DB
Tsai, Manish Amde, Sean Owen, Doris Xin, Reynold Xin, Michael J. Franklin, Reza Zadeh, Matei Zaharia, and Ameet
Talwalkar. 2016. MLlib: Machine Learning in Apache Spark. _Journal of Machine Learning Research_ 17, 34 (2016), 1–7.

[29] Roger B Myerson. 1981. Optimal auction design. _Mathematics of operations research_ 6, 1 (1981), 58–73.

[30] Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu
Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, Jake Vanderplas, Alexandre Passos, David Cournapeau,
Matthieu Brucher, Matthieu Perrot, and Édouard Duchesnay. 2011. Scikit-learn: Machine Learning in Python. _J. Mach._
_Learn. Res._ [12 (Nov. 2011), 2825–2830. http://dl.acm.org/citation.cfm?id=1953048.2078195](http://dl.acm.org/citation.cfm?id=1953048.2078195)

[31] John G Riley and William F Samuelson. 1981. Optimal Auctions. _The American Economic Review_ 71, 3 (1981), 381–392.

[32] Jean-Charles Rochet and Jean Tirole. 2003. Platform Competition in Two-Sided Markets. _Journal of the European_
_Economic Association_ 1, 4 (2003), 990–1029.

[33] Ludwig Schmidt, Shibani Santurkar, Dimitris Tsipras, Kunal Talwar, and Aleksander Madry. 2018. Adversarially Robust
Generalization Requires More Data. In _Advances in Neural Information Processing Systems 31_ . Curran Associates, Inc.,
[5019–5031. http://papers.nips.cc/paper/7749-adversarially-robust-generalization-requires-more-data.pdf](http://papers.nips.cc/paper/7749-adversarially-robust-generalization-requires-more-data.pdf)

[34] LS Shapley. 1952. _A VALUE FOR N-PERSON GAMES_ . Technical Report. RAND Corp Santa Monica CA.

[35] Hal R Varian. 2009. Online Ad Auctions. _American Economic Review_ 99, 2 (2009), 430–34.

[36] Justin Wolfers and Eric Zitzewitz. 2004. Prediction markets. _Journal of economic perspectives_ 18, 2 (2004), 107–126.

[37] Weinan Zhang, Shuai Yuan, and Jun Wang. 2014. Optimal Real-time Bidding for Display Advertising. In _Proceedings of_
_the 20th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD ’14)_ . ACM, New York,
[NY, USA, 1077–1086. https://doi.org/10.1145/2623330.2623633](https://doi.org/10.1145/2623330.2623633)


**A** **TRUTHFULNESS**


Theorem A.1 ( **Theorem 5.3** ). _For_ AF [∗] _, Property 3.1 (Truthfulness) can be achieved if and only_
_if Assumption 1 holds. In which case,_ RF [∗] _guarantees truthfulness._


Proof. This is a classic result from [ 29 ]. We provide the arguments here for completeness and
for consistency with the properties and notation we introduce in our work. We begin with the
backward direction. By Assumption 1 the following then holds ∀ _b_ _n_ [′] [≥] _[b]_ _[n]_


G( _Y_ _n_, M(AF [∗] ( _b_ _n_ [′] [,] _[p]_ _[n]_ [))) ≥G(] _[Y]_ _[n]_ [,][ M(AF] [ ∗] [(] _[b]_ _[n]_ [,] _[p]_ _[n]_ [)))] (4)


To simplify notation, let _h_ ( _z_ ; G, _p_ _n_, _Y_ _n_, M) = G( _Y_ _n_, M(AF [∗] ( _z_, _p_ _n_ ))) . In words, _h_ ( _z_ ) is the gain
in prediction accuracy as a function of the bid, _z_, for a fixed G, _Y_ _n_, M, _p_ _n_ .
By definition of (1), it suffices to show that if _b_ _n_ � _µ_ _n_, the following holds



_µ_ _n_ _b_ _n_

_h_ ( _z_ ) _dz_ ≥ _µ_ _n_  - _h_ ( _b_ _n_ ) − _b_ _n_  - _h_ ( _b_ _n_ ) +
0 ∫ 0



_µ_ _n_
_µ_ _n_ - _h_ ( _µ_ _n_ ) − _µ_ _n_ - _h_ ( _µ_ _n_ ) +
∫ 0



_h_ ( _z_ ) _dz_ (5)

0



This is equivalent to showing that

_µ_ _n_

_h_ (

∫ 0



_h_ ( _z_ ) _dz_ −( _b_ _n_ − _µ_ _n_ ) · _h_ ( _b_ _n_ ) (6)
0



_µ_ _n_ _b_ _n_

_h_ ( _z_ ) _dz_ ≥
0 ∫ 0



Case 1: _b_ _n_  - _µ_ _n_ . In this case, (6) is equivalent to


_b_ _n_
( _b_ _n_ − _µ_ _n_ ) · _h_ ( _b_ _n_ ) ≥ _h_ ( _z_ ) _dz_ (7)
∫ _µ_ _n_


This is immediately true due to monotonicity of _h_ ( _z_ ) which comes from (4) . Case 2: _b_ _n_ < _µ_ _n_ . In this
case, (6) is equivalent to

_µ_ _n_

_h_ ( _z_ ) _dz_ ≥( _µ_ _n_ − _b_ _n_ ) · _h_ ( _b_ _n_ ) (8)

∫ _b_ _n_


Again, this is immediately true due to monotonicity of _h_ ( _z_ ).


Now we prove the opposite direction, i.e. if we have a truthful payment mechanism, which we
denote as RF [′], an increased allocation of features cannot decrease accuracy. Our definition of a
truthful payment function implies the following two inequalities ∀ _b_ - _a_


_a_            - _h_ ( _a_ ) −RF [′] (·, _a_, ·) ≥ _a_            - _h_ ( _b_ ) −RF [′] (·, _b_, ·) (9)


_b_            - _h_ ( _b_ ) −RF [′] (·, _b_, ·) ≥ _b_            - _h_ ( _a_ ) −RF [′] (·, _a_, ·) (10)


These two inequalities imply


_a_       - _h_ ( _a_ ) + _b_       - _h_ ( _b_ ) ≥ _a_       - _h_ ( _b_ ) + _b_       - _h_ ( _a_ ) =⇒ _h_ ( _b_ )( _b_ − _a_ ) ≥ _h_ ( _a_ )( _b_ − _a_ ) (11)


Since by construction _b_ − _a_ - 0, we can divide both sides of the inequality by _b_ − _a_ to get


_h_ ( _b_ ) ≥ _h_ ( _a_ ) ⇐⇒G _n_ ( _Y_ _n_, M(AF [∗] ( _b_, _p_ _n_ ))) ≥G _n_ ( _Y_ _n_, M(AF [∗] ( _a_, _p_ _n_ ))) (12)


Since the allocation function AF [∗] ( _z_, _p_ _n_ ) is increasing in _z_, this completes the proof. 

**B** **PRICE UPDATE - PROOF OF THEOREM 5.2**

Theorem B.1 ( **Theorem 5.2** ). _Let Assumptions 1, 3 and 4 hold. Let_ _p_ _n_ : _n_ ∈[ _N_ ] _be the output of_
_Algorithm 1. Let_ L _be the Lipschitz constant of_ RF [∗] _with respect to price (where_ L _is defined as in_
_Assumption 3). Let_ B max ∈ R _be the maximum element of_ B _(where_ B _is defined as in Assumption 4)._



1
_Then by choosing algorithm hyper-parameters_ _ϵ_ =
L ~~√~~



_N_ [,] _[ δ]_ [ =] ~~�~~



log(| B _net_ ( _ϵ_ )|)



_net_

_Then by choosing algorithm hyper-parameters_ _ϵ_ = L ~~√~~ _N_ [,] _[ δ]_ [ =] _N_ _for some positive constant_

_C_ - 0 _, the total average regret is bounded by,_



�



log( _N_ )

).
_N_



1
_N_ [E][[R(] _[N]_ [)] ≤] _[C]_ [B] [max]



�



log(B max L ~~√~~



L ~~√~~ _N_ )

= _O_ (
_N_



_where the expectation is taken over the randomness in Algorithm 1. Hence, Property 3.2 (Revenue_
_Maxmization) holds._


Proof. Our proof here is an adaptation of the classic result from [ 3 ]. We provide the arguments
here for completeness and for consistency with the properties and notation we introduce in our
work. It is easily seen by Assumption 1 that the revenue function RF [∗] is non-negative. Now since
by construction the gain function, G ∈[ 0, 1 ], the range of RF [∗] is in [ 0, B max ] . This directly implies
that for all _i_ and _n_, _д_ _n_ _[i]_ [∈[] [0] [,] [ 1] []] [ (recall] _[ д]_ _n_ _[i]_ [is the (normalized) revenue gain if we played price] _[ i]_ [ for]
every buyer _n_ ).
We first prove a regret bound for the best fixed price in hindsight within B net ( _ϵ_ ) . Let _д_ _n_ [alg] be the
expected (normalized) gain of Algorithm 1 for buyer _n_ . By construction,


|B net ( _ϵ_ )|
� _i_ =1 _w_ _n_ _[i]_ _[д]_ _n_ _[i]_
_д_ _n_ [alg] =
_W_ _n_

Observe we have the following inductive relationship regarding _W_ _n_



_W_ _n_ +1 =


=



|B net ( _ϵ_ )|
� _w_ _n_ _[i]_ +1 (13)

_i_ =1


|B net ( _ϵ_ )|
� _w_ _n_ _[i]_ [+] _[ δw]_ _n_ _[i]_ _[д]_ _n_ _[i]_ (14)

_i_ =1



= _W_ _n_ + _δ_



|B net ( _ϵ_ )|
� _w_ _n_ _[i]_ _[д]_ _n_ _[i]_ (15)

_i_ =1



= _W_ _n_ (1 + _δд_ _n_ [alg] [)] (16)

= _W_ 1 Π _[n]_ _i_ =1 [(][1][ +] _[ δд]_ _n_ [alg] [)] (17)

( = _a_ ) |B net ( _ϵ_ )| · Π _ni_ =1 [(][1][ +] _[ δд]_ _n_ [alg] [)] (18)


where (a) follows since _W_ 1 was initialized to be |B net ( _ϵ_ )|.
Taking logs and utilizing the inequality log(1 + _x_ ) ≤ _x_ for _x_ ≥ 0, we have



log( _W_ _N_ +1 ) = log(|B net ( _ϵ_ )|) +


≤ log(|B net ( _ϵ_ )|) +



_N_
� log(1 + _δд_ _n_ [alg] [))] (19)

_i_ =1


_N_
� _δд_ _n_ [alg] (20)

_i_ =1


Now using that log(1 + _x_ ) ≥ _x_ − _x_ [2] for _x_ ≥ 0, we have for all prices _c_ _[i]_ ∈B net ( _ϵ_ ),


log( _W_ _N_ +1 ) ≥ log( _w_ _n_ _[i]_ +1 [)] (21)



_N_
� log(1 + _δд_ _n_ _[i]_ [))] (22)

_n_ =1


_N_
� _δд_ _n_ _[i]_ [−(] _[δд]_ _n_ _[i]_ [)] [2] (23)

_n_ =1


_N_
� _δд_ _n_ _[i]_ [−] _[δ]_ [2] _[N]_ (24)

_i_ =1



=


≥



( _a_ )
≥



where (a) follows since _д_ _n_ _[i]_ [∈[][0][,][ 1][]][.]
Thus for all prices _c_ _[i]_ ∈B net ( _ϵ_ )


_N_
� _δд_ _n_ [alg] ≥

_n_ =1



_N_
� _δд_ _n_ _[i]_ [−] [log][(|B] [net] [(] _[ϵ]_ [)|) −] _[δ]_ [2] _[N]_

_n_ =1



Dividing by _δN_ and picking _δ_ = �



log(| B net ( _ϵ_ )|)

_N_, we have for all prices _c_ _[i]_ ∈B net ( _ϵ_ )



_N_



�



1

_N_



_N_
�



� _д_ _n_ [alg] ≥ _N_ [1]

_n_ =1



log(|B net ( _ϵ_ )|)


_N_



_N_
�



� _д_ _n_ _[i]_ [−] [2]

_i_ =1



So far we have a bound on how well Algorithm 1 performs against prices in B net ( _ϵ_ ) . We now
extend it to all of B . Let _д_ _n_ [opt] be the (normalized) revenue gain from buyer _n_ if we had played the
optimal price, _p_ [∗] (as defined in Property 3.2). Note that by Assumption 4, we have _p_ [∗] ∈B . Then by
the construction of |B net ( _ϵ_ )|, there exists _c_ _[i]_ ∈B net ( _ϵ_ ) such that | _c_ _[i]_ − _p_ [∗] | ≤ _ϵ_ . Then by Assumption
3, we have that


1 L _ϵ_
| _д_ _n_ [opt] − _д_ _n_ _[i]_ [|][ =] |RF [∗] ( _p_ [∗], _b_ _n_, _Y_ _n_ ) −RF [∗] ( _c_ _[i]_, _b_ _n_, _Y_ _n_ )| ≤
B max B max


We thus have



_N_



B max



1

_N_



_N_
�



� _д_ _n_ [alg] ≥ _N_ [1]

_n_ =1



log(|B net ( _ϵ_ )|)



_N_
� _д_ _n_ [opt] − 2

_i_ =1



~~�~~



net ( _ϵ_ )|) − [L] _[ϵ]_

_N_ B



Multiplying throughout by B max, we get



_N_ [RF] [ ∗] [(] _[p]_ [∗] [,] _[b]_ _[n]_ [,] _[Y]_ _[n]_ [) −] [2][B] [max]



1

_N_



_N_
�



� E[RF [∗] ( _p_ _n_, _b_ _n_, _Y_ _n_ )] ≥ _N_ [1]

_n_ =1



�



log(|B net ( _ϵ_ )|)

−L _ϵ_
_N_



log(|B net ( _ϵ_ )|)



1
Now setting _ϵ_ =
L ~~√~~



_N_ [and noting that] [ |B] [net] [(] _[ϵ]_ [)| ≤] [3][B] _ϵ_ [max], for some positive constant _C_ - 0, we have



_N_ [RF] [ ∗] [(] _[p]_ [∗] [,] _[b]_ _[n]_ [,] _[Y]_ _[n]_ [) −] _[C]_ [B] [max]



~~�~~



1

_N_



_N_
�



� E[RF [∗] ( _p_ _n_, _b_ _n_, _Y_ _n_ )] ≥ _N_ [1]

_n_ =1



log(B max L ~~√~~ _N_ )


_N_



log(B max L ~~√~~






**C** **FAIRNESS**


Theorem C.1 ( **Theorem 5.3** ). _Let_ _ψ_ _n_, _shapley_ _be the unique vector satisfying Property 3.3 as given_
_in_ (2) _. For Algorithm 2, pick the following hyperparameter:_ _K_ - _[M]_ [ lo] 2 [g] _ϵ_ [(] [2] [2][/] _[δ]_ [)] _, where_ _δ_, _ϵ_ - 0 _. Then with_

_probability_ 1 − _δ_ _, the output_ _ψ_ [ˆ] _n_ _of Algorithm 2, achieves the following_


∥ _ψ_ _n_, _shapley_ − _ψ_ [ˆ] _n_ ∥ ∞ < _ϵ_ (25)


Proof. It is easily seen that _ψ_ _n_,shapley can be formulated as the following expectation


_ψ_ _n_,shapley ( _m_ ) = E _σ_ ∼Unif( _σ_ _Sn_ ) [G _n_ ( _Y_ _n_, M _n_ ( _X_ [ _σ_ < _m_ ∪ _m_ ] )) −G _n_ ( _Y_ _n_, M _n_ ( _X_ [ _σ_ < _m_ ] )] (26)


The random variable _ψ_ [ˆ] _n_ _[k]_ [(] _[m]_ [)][ is distributed in the following manner:]


ˆ
P _ψ_ _n_ _[k]_ [(] _[m]_ [)][ =][ G] _[n]_ [(] _[Y]_ _[n]_ [,][ M(] _[X]_ [ [] _[σ]_ _k_ [<] _[m]_ [ ∪] _[m]_ []] [)) −G] _[n]_ [(] _[Y]_ _[n]_ [,][ M(] _[X]_ [ [] _[σ]_ _k_ [<] _[m]_ []] [))][;] _[σ]_ [ ∈] _[σ]_ _[S]_ _n_ = [1] (27)
� � _S_ _n_ !


We then have



E[ _ψ_ [ˆ] _n_ ( _m_ )] = [1]

_K_



_K_
� E[ _ψ_ [ˆ] _n_ _[k]_ [(] _[m]_ [)]][ =] _[ ψ]_ _[n]_ [,][shapley] (28)

_k_ =1



Since _ψ_ [ˆ] _n_ ( _m_ ) has bounded support between 0 and 1, and the _ψ_ [ˆ] _n_ _[k]_ [(] _[m]_ [)] [ are i.i.d, we can apply Hoeffd-]
ing’s inequality to get the following bound

P | _ψ_ _n_,shapley − _ψ_ [ˆ] _n_ ( _m_ )| > _ϵ_ < 2 exp( [−][2] _[ϵ]_ [2] ) (29)
� � _K_


By applying a Union bound over all _m_ ∈ _S_ _n_ ≤ _M_, we have

P ∥ _ψ_ _n_,shapley − _ψ_ [ˆ] _n_ ∥ ∞           - _ϵ_ < 2 _M_ exp( [−][2] _[ϵ]_ [2] ) (30)
� � _K_



Setting _δ_ = 2 _M_ exp( [−][2] _K_ _[ϵ]_ [2] [)][ and solving for] _[ K]_ [ completes the proof.] 


Theorem C.2 ( **Theorem 5.4** ). _Let Assumption 2 hold. For Algorithm 3, pick the following hyper-_
_parameters:_ _K_ ≥ _[M]_ [ lo] 2( [g] ~~_[ϵ]_~~ 3 [(][2][)] [2] [/] _[δ]_ [)], _λ_ = log( 2 ) _, where_ _δ_, _ϵ_ - 0 _. Then with probability_ 1 − _δ_ _, the output,_ _ψ_ _n_ _, of_

_Algorithm 3 is_ _ϵ_ _-“Robust to Replication" i.e. Property 3.4 (Robustness-to-Replication) holds. Additionally_
_Conditions 2-4 of Property 3.3 continue to hold for ψ_ _n_ _with ϵ-precision._


Proof. To reduce notational overhead, we drop the dependence on _n_ of all variables for the
remainder of the proof. Let _S_ = { _X_ 1, _X_ 2, . . ., _X_ _K_ } refer to the original set of allocated features
without replication. Let _S_ [+] = { _X_ (1,1), _X_ (1,2), . . ., _X_ (1, _c_ 1 ), _X_ (2,1), . . ., _X_ ( _K_, _c_ _K_ ) } (with _c_ _i_ ∈ N ), be an
appended version of _S_ with replicated versions of the original features, i.e. _X_ ( _m_, _i_ ) is the ( _i_ − 1 ) -th
replicated copy of feature _X_ _m_ .
Let _ψ_ [ˆ], _ψ_ [ˆ] [+] be the respective outputs of Step 1 of Algorithm 3 for _S_, _S_ [+] respectively. The total
revenue allocation to seller _m_ in the original and replicated setting is given by the following:


_ψ_ ( _m_ ) = _ψ_ [ˆ] ( _m_ ) exp(− _λ_ � SM( _X_ _m_, _X_ _j_ )) (31)

_j_ ∈ _S_ _m_ \{ _m_ }



� _ψ_ ˆ [+] [�] ( _m_, _i_ )� exp(− _λ_ �

_i_ ∈ _c_ _m_ ( _k_ )∈ _S_ [+]



_ψ_ [+] ( _m_ ) = �



� SM( _X_ ( _m_, _i_ ), _X_ ( _j_, _k_ ) )) (32)

( _j_, _k_ )∈ _S_ _m_ [+] \{( _m_, _i_ )}


For Property 3.4 to hold, it suffices to show that _ψ_ [+] ( _m_ ) ≤ _ψ_ ( _m_ ) + _ϵ_ . We have that

� _ψ_ ˆ [+] [�] ( _m_, _i_ )� exp(− _λ_ � SM( _X_ ( _m_, _i_ ), _X_ ( _j_, _k_ ) ))



� _ψ_ ˆ [+] [�] ( _m_, _i_ )� exp(− _λ_ �

_i_ ∈ _c_ _m_ ( _k_ )∈ _S_ [+]



� SM( _X_ ( _m_, _i_ ), _X_ ( _j_, _k_ ) ))

( _j_, _k_ )∈ _S_ _m_ [+] \{( _m_, _i_ )}



( _a_ )
≤
�



� SM( _X_ ( _m_, _i_ ), _X_ ( _m_, _j_ ) )) exp(− _λ_ �

_j_ ∈[ _c_ _m_ ]\ _i_ _l_ ∈ _S_ _m_ \{



� _ψ_ ˆ [+] [�] ( _m_, _i_ )� exp(− _λ_ �

_i_ ∈ _c_ _m_ _j_ ∈[ _c_



� SM( _X_ ( _m_, _i_ ), _X_ ( _l_,1) ))

_l_ ∈ _S_ _m_ \{ _m_ }



( _b_ )
=
�



� SM( _X_ ( _m_, _i_ ), _X_ ( _l_,1) ))

_l_ ∈ _S_ _m_ \{ _m_ }



� _ψ_ ˆ [+] [�] ( _m_, _i_ )� exp(− _λ_ ( _c_ _m_ − 1)) exp(− _λ_ �

_i_ ∈ _c_ _m_ _l_ ∈ _S_



� SM( _X_ _m_, _X_ _j_ ))

_j_ ∈ _S_ _m_ \{ _m_ }



( _c_ )
≤ _c_ _m_


≤ _c_ _m_



_ψ_ ˆ [+] [�] ( _m_, 1) + [1]
� � 3


ˆ
_ψ_ [+] [�] ( _m_, 1) + [1]
� � 3



3 [1] _[ϵ]_ � exp(− _λ_ ( _c_ _m_ − 1)) exp(− _λ_ _l_ ∈ _S_ � _m_ \{

[1] 3 _[ϵ]_ � exp(− _λ_ ( _c_ _m_ − 1)) exp(− _λ_ �



� SM( _X_ ( _m_,1), _X_ ( _l_,1) ))

_l_ ∈ _S_ _m_ \{ _m_ }



(a) follows since _λ_, SM(·) ≥ 0; (b) follows by condition (i) of Definition 4.1; (c) follows from
Theorem 5.3;



Hence it suffices to show that _c_ _m_


have



_ψ_ ˆ [+] [�] ( _m_, 1 )� + [1] 3 _[ϵ]_ exp(− _λ_ ( _c_ _m_ − 1 )) ≤ _ψ_ [ˆ] ( _m_ ) + _ϵ_ ∀ _c_ _m_ ∈ N . We
� �



ˆ
_c_ _m_ exp(− _λ_ ( _c_ _m_ − 1)) _ψ_ [+] [�] ( _m_, 1) + [1]
� � 3




[1] ( ≤ _d_ ) _c_ _m_ exp(− _λ_ ( _c_ _m_ − 1)) _ψ_ ( _m_ )

3 _[ϵ]_ � � _c_ _m_



( _m_ )

+ [2]
_c_ _m_ 3



3 _[ϵ]_ �



( _e_ )
≤ _c_ _m_ exp(− _λ_ ( _c_ _m_ − 1)) _ψ_ ( _m_ ) + [2]
� 3 _[ϵ]_ �



( _f_ ) ˆ
≤ _c_ _m_ exp(− _λ_ ( _c_ _m_ − 1)) _ψ_ ( _m_ ) + _ϵ_
� �


( _д_ ) ˆ
≤ _ψ_ ( _m_ ) + _ϵ_
� �


where (d) and (f) follow from Theorem 5.3; (e) follows since _c_ _m_ ∈ N ; (g) follows since _c_ _m_ exp(− _λ_ ( _c_ _m_ −
1)) ≤ 1 ∀ _c_ _m_ ∈ N by picking _λ_ = log(2).
The fact that Conditions 2-4 of Property 3.3 continue to hold for follow _ψ_ _n_ with _ϵ_ -precision
follow easily from Theorem 5.3 and the construction of _ψ_ _n_ . 

Proposition C.1 ( **Proposition 5.1** ). _If the identities of sellers in the marketplace is anonymized,_
_the balance condition in Property 3.3 and Property 3.4 cannot simultaneously hold._



Proof. We show this through an extremely simple counter-example consisting of three scenarios.
In the first scenario, the marketplace consists of exactly two sellers, _A_, _B_, each selling identical
features i.e. _X_ _A_ = _X_ _B_ . By Condition 1 and 2 of Property 3.3, both sellers must receive an equal
allocation i.e. _ψ_ 1 ( _A_ ) = _ψ_ 1 ( _B_ ) = [1] 2 [for any prediction task.]

Now consider a second scenario, where the marketplace against consists of the same two sellers,
_A_ and _B_, but this time seller _A_ replicates his or her feature once and sells it again in the marketplace
as _A_ [′] . Since by assumption the identity of sellers is anonymized, to achieve the “balance" condition
in Property 3.3, we require _ψ_ 2 ( _A_ ) = _ψ_ 2 ( _B_ ) = _ψ_ 2 ( _A_ [′] ) = [1] 3 [. Thus the total allocation to seller] _[ A]_ [ is]

_ψ_ 2 ( _A_ ) + _ψ_ 2 ( _A_ [′] ) = [2] [>] [1] [=] _[ ψ]_ [1] [(] _[A]_ [)][ i.e. Property 3.4 does not hold.]




[2] [1]

3 [>] 2



2 [=] _[ ψ]_ [1] [(] _[A]_ [)][ i.e. Property 3.4 does not hold.]


Finally consider a third scenario, where the marketplace consists of three sellers _A_, _B_ and _C_, each
selling identical features i.e. _X_ _A_ = _X_ _B_ = _X_ _C_ . It is easily seen that to achieve “balance", we require
_ψ_ 3 ( _A_ ) = _ψ_ 3 ( _B_ ) = _ψ_ 3 ( _C_ ) = 3 [1] [.]

Since the marketplace cannot differentiate between _A_ [′] and _C_, we either have balance or Property
3.4 i.e. “robustness to replication". 

**D** **EFFICIENCY**

Corollary D.1 ( **Corollary 5.1** ). AF [∗], RF [∗], PF [∗] _run in_ _O_ ( _M_ ) _._ PD _a_ [∗] _[,]_ [ PD] _b_ [∗] _[run in]_ _[ O]_ [(] _[M]_ [2] [)]
_time. Hence, Property 3.5 holds._


Proof. This is immediately seen by studying the four functions: (i) AF [∗] simply tunes the quality
of each feature _X_ _j_ for _j_ ∈[ _M_ ], which is a linear time operation in _M_ ; (ii) RF [∗] again runs in linear
time as we require a constant number of calls to G and M ; (iii) PF [∗] runs in linear time as we call
G and M once for every price in B net ( _ϵ_ ) ; (iv) PD _a_ [∗] [has a running time of] _[M]_ [2] [ lo] 2 _ϵ_ [g][(] [2] [2][/] _[δ]_ [)] for any level

of precision and confidence given by _ϵ_, _δ_ respectively i.e. we require _[M]_ [ lo] 2 [g] _ϵ_ [(] [2] [2][/] _[δ]_ [)] calls to G and M to

compute the Shapley Allocation for each feature _X_ _j_ for _j_ ∈[ _M_ ] . The additional step in PD _b_ [∗] [i.e.]
Step 2, is also a linear time operation in _M_ (note that the pairwise similarities between _X_ _i_, _X_ _j_ for
any _i_, _j_ ∈[ _M_ ] can be precomputed). 

**E** **OPTIMAL BALANCE-PRESERVING, ROBUST-TO-REPLICATION PENALTY**

**FUNCTIONS**


In this section we provide a necessary and sufficient condition for “robustness-to-replication" any
penalty function _f_ : R + → R + must satisfy, where _f_ takes as argument the cumulative similarity
of a feature with all other features. In Algorithm 3, we provide a specific example of such a penalty
function given by exponential down-weighting. We have the following result holds


Proposition E.1 ( **Proposition 5.2** ). _Let Assumption 2 hold. Then for a given similarity metric_
SM _, a penalty function f is “robust-to-replication" if and only if it satisfies the following relation_


( _c_ + 1) _f_ ( _x_ + _c_ ) ≤ _f_ ( _x_ )


_where c_ ∈ Z +, _x_ ∈ R + _._


Proof. Consider the case where a certain data seller with feature _X_ _i_ has original cumulative
similarity _x_, and makes _c_ additional copies of its own data. The following relation is both necessary
and sufficient to ensure robustness,


ˆ
_ψ_ _i_ ( _c_ + 1) _f_ ( _x_ + _c_ ) ≤ _ψ_ _i_ _f_ ( _x_ )


We first show sufficiency. By Assumption 2, the new Shapley value (including the replicated
features) for a single feature _X_ _i_ denoted by _ψ_ [ˆ], is no larger than the original Shapley value, _ψ_, for
the same feature. Then it immediately follows that ( _c_ + 1) _f_ ( _x_ + _c_ ) ≤ _f_ ( _x_ ).
We now show that it is also necessary. We study how much the Shapley allocation changes when
only one player duplicates data. The Shapley allocation for feature _X_ _i_ is defined as



_ψ_ _i_ ( _v_ ) = �

_S_ ⊆ _N_ \{ _i_ }



| _S_ |!(| _N_ | − | _S_ | − 1)!

( _v_ ( _S_ ∪{ _i_ }) − _v_ ( _S_ ))
| _N_ |!



A key observation to computing the new Shapley value is that _v_ ( _S_ ∪{ _i_ }) − _v_ ( _S_ ) ≥ 0 if _i_ appears
before all its copies. Define _M_ to be the number of original sellers (without copying) and _c_ are the


additional copies. By a counting argument one can show that




[ _v_ ( _S_ ∪{ _i_ }) − _v_ ( _S_ )]
�


1
� _M_ ! [[] _[v]_ [(] _[S]_ [ ∪{] _[i]_ [}) −] _[v]_ [(] _[S]_ [)]]


1
� _M_ ! [[] _[v]_ [(] _[S]_ [ ∪{] _[i]_ [}) −] _[v]_ [(] _[S]_ [)]]



_M_ − _i_ + _c_ − 1
� _M_ − _i_ − 1


_M_ − _i_ + _c_ − 1
� _M_ − _i_ − 1


_M_ − _i_ + _c_ − 1
� _c_



ˆ
_ψ_ _i_ ( _v_ ) =


=


=



_M_ −1
�

_i_ =0


_M_ −1
�

_i_ =0


_M_ −1
�

_i_ =0



1

( _M_ + _c_ )!


_M_ !

( _M_ + _c_ )!


_M_ !

( _M_ + _c_ )!



_M_
≤
_M_ + _c_



_M_ −1
�

_i_ =0



1
_M_ ! [[] _[v]_ [(] _[S]_ [ ∪{] _[i]_ [}) −] _[v]_ [(] _[S]_ [)]]



_M_

=
_M_ + _c_ _[ψ]_ _[i]_ [(] _[v]_ [)]


Observe this inequality turns into an equality when all the original sellers have exactly the same
data. We observe that for a large number of unique sellers then copying does not change the Shapley
allocation too much ≃− _c_ / _M_ . In fact, this bound tells us that when there are a large number of
sellers, replicating a single data set a fixed number of times does not change the Shapley allocation
too much, _i.e._, _ψ_ [ˆ] _i_ ≈ _ψ_ [ˆ] _i_ (with the approximation being tight in the limit as _M_ tends to infinity).
Therefore, we necessarily need to ensure that


( _c_ + 1) _f_ ( _x_ + _c_ ) ≤ _f_ ( _x_ )


                               

Remark E.1. _If we make the_ extremely loose _relaxation of letting_ _c_ ∈ R + _instead of_ Z + _, then the_
_exponential weighting in Algorithm 3 is minimal in the sense that it ensures robustness with least_
_penalty in allocation. Observe that the penalty function (assuming differentiability) should also satisfy_

_f_ ( _c_ + _x_ ) − _f_ ( _x_ )

≤− _f_ ( _x_ + _c_ )
_c_

_f_ ( _c_ + _x_ ) − _f_ ( _x_ )
lim ≤− _f_ ( _x_ )
_c_ →0 [+] _c_


′
_f_ ( _x_ ) ≤− _f_ ( _x_ )


_By Gronwall’s Inequality we can see that_ _f_ ( _x_ ) ≤ _Ce_ [−] _[Kx]_ _for suitable_ _C_, _K_ ≥ 0 _. This suggests that the_
_exponential class of penalty ensure robustness with the “least” penalty, and are minimal in that sense._


