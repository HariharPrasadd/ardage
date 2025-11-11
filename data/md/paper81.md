# **Nonparametric causal effects based on** **incremental propensity score interventions**

#### Edward H. Kennedy [∗] Department of Statistics, Carnegie Mellon University June 20, 2018

**Abstract**


Most work in causal inference considers deterministic interventions that set each

unit’s treatment to some fixed value. However, under positivity violations these interventions can lead to non-identification, inefficiency, and effects with little practical
relevance. Further, corresponding effects in longitudinal studies are highly sensitive
to the curse of dimensionality, resulting in widespread use of unrealistic parametric
models. We propose a novel solution to these problems: incremental interventions
that shift propensity score values rather than set treatments to fixed values. Incremental interventions have several crucial advantages. First, they avoid positivity
assumptions entirely. Second, they require no parametric assumptions and yet still
admit a simple characterization of longitudinal effects, independent of the number
of timepoints. For example, they allow longitudinal effects to be visualized with a
single curve instead of lists of coefficients. After characterizing incremental interventions and giving identifying conditions for corresponding effects, we also develop
general efficiency theory, propose efficient nonparametric estimators that can attain
fast convergence rates even when incorporating flexible machine learning, and propose a bootstrap-based confidence band and simultaneous test of no treatment effect.
Finally we explore finite-sample performance via simulation, and apply the methods
to study time-varying sociological effects of incarceration on entry into marriage.


_Keywords:_ observational study, positivity, stochastic intervention, time-varying confounding, treatment effect.


_∗_ Edward Kennedy is Assistant Professor in the Department of Statistics, Carnegie Mellon University,
Pittsburgh, PA 15213 (e-mail: edward@stat.cmu.edu). The author thanks Traci Kennedy, Miguel Hernan,
Kwangho Kim, and the Causal Inference Reading Group at Carnegie Mellon for helpful discussions and
comments, and Valerio Bacak for guidance on the National Longitudinal Survey of Youth data analysis.


### **1 Introduction**

Most work in causal inference considers deterministic interventions that set each unit’s

treatment to some fixed value. For example, the usual average treatment effect indicates
how mean outcomes would change if all units were uniformly assigned treatment versus
control. Similarly, standard marginal structural models (Robins _et al._ 2000) describe outcomes had all units followed given exposure trajectories over time (e.g., treated at every
time, treated after time _t_, etc.). However, these simple effects are not identified when
some units have zero chance to receive given treatment options. This is a violation of the
so-called positivity assumption, which has been known in the causal inference literature
since at least Rosenbaum & Rubin (1983). Even if positivity is only nearly violated (i.e.,
chances of some treatment options are merely small), the finite-sample behavior of many
common estimators can be severely degraded (Kang & Schafer 2007; Moore _et al._ 2012).
Similarly, even if positivity holds, in longitudinal studies these standard effects are afflicted by a curse of dimensionality in the number of study timepoints: exponentially many
samples are needed to learn about all treatment trajectories. For example, in a simple
trial with a binary randomized treatment and ten timepoints, we would need nearly 12,000
patients to guarantee _<_ 1% chance of having any unrepresented exposure trajectories. The
usual way to deal with this problem is to assume it away with a parametric model for how
outcomes change across trajectories. However, such models are typically severely wrong
if overly simple, and can be hard to interpret otherwise (and are often still misspecified).
Further, in the real world, treatment is typically not applied uniformly, so static deterministic interventions may not be of practical policy interest. For example, most medical
treatments would never be applied indiscriminately, but instead would be recommended or
not based on characteristics of the patient and physician prescribing preference.
Thus there has been substantial recent interest in dynamic and stochastic interventions,
which can depend on unit characteristics and be random rather than deterministic. Examples have been studied for point exposures by Dud´ık _et al._ (2014), Pearl (2009), and Tian
(2008), and in longitudinal studies by Cain _et al._ (2010), Murphy _et al._ (2001), Robins _et al._
(2004, 2008), Taubman _et al._ (2009), van der Laan & Petersen (2007), and Young _et al._
(2011). Particularly relevant to this paper is work by D´ıaz & van der Laan (2013, 2012),
Haneuse & Rotnitzky (2013), Moore _et al._ (2012), and Young _et al._ (2014), who consider
interventions that depend on the observational treatment process. However, to the best of
our knowledge, none of the existing intervention effects both avoids positivity conditions
entirely and is completely nonparametric (even in studies with many timepoints).
In this paper we propose novel _incremental intervention effects_, based on shifting
propensity scores rather than setting treatment values. We show that such effects can
be identified and estimated without any positivity or parametric assumptions, and argue
that they can be more realistic than other interventions. One trade-off is that they yield
effects that are more descriptive than prescriptive. We develop nonparametric influencefunction-based estimators that can incorporate flexible machine learning tools, while still
providing valid parametric-rate inference. Our methods for uniform inference (using the
multiplier bootstrap) also yield a new general test of no treatment effect. We conclude with
a simulation study, and apply the methods in a longitudinal study of incarceration effects.


1


### **2 Notation & Setup**

We consider the case where we observe a sample ( **Z** 1 _, ...,_ **Z** _n_ ) of iid observations from distribution P, with
**Z** = ( **X** 1 _, A_ 1 _,_ **X** 2 _, A_ 2 _, ...,_ **X** _T_ _, A_ _T_ _, Y_ )


for covariates **X** _t_ and treatment _A_ _t_ at time _t_, and outcome _Y_ . For simplicity, at present we
consider binary treatments (so the support of _A_ _t_ is _A_ = _{_ 0 _,_ 1 _}_ ) and completely observed
**Z** (so there is no missingness or dropout), but extensions will appear in future work.
We use overbars to denote the past history of a variable, so that **X** _t_ = ( **X** 1 _, ...,_ **X** _t_ ) and
_A_ _t_ = ( _A_ 1 _, ..., A_ _t_ ), for example, and we let **H** _t_ = ( **X** _t_ _, A_ _t−_ 1 ) denote the past history just
prior to treatment at time _t_, with support _H_ _t_ .

_Remark_ 1 _._ The above data setup also covers the case where outcomes are time-varying,
i.e., where rather than **Z** the observations are given by **Z** _[∗]_ = ( **X** _[∗]_ 1 _[, A]_ [1] _[, Y]_ [1] _[, ...,]_ **[ X]** _[∗]_ _T_ _[∗]_ _[, A]_ _[T]_ _[ ∗]_ _[, Y]_ _[T]_ _[ ∗]_ [).]
This follows since we can let **X** _t_ = ( **X** _[∗]_ _t_ _[, Y]_ _[t][−]_ [1] [) and] _[ Y]_ [ =] _[ Y]_ _[T]_ _[ ∗]_ [in our original formulation. If]
interest centers on treatment effects on an earlier outcome _Y_ _t_ (for _t < T_ _[∗]_ ), rather than _Y_ _T_ _∗_,
then we can let _Y_ = _Y_ _t_ and truncate the sequence, defining _T_ = _t_ instead of _T_ = _T_ _[∗]_ .
In this paper we use potential outcomes (Rubin 1974), and so let _Y_ ~~_[a]_~~ _[T]_ denote the outcome that would have been observed had the treatment sequence _A_ _T_ = ~~_a_~~ _T_ been received.
The quantity _Y_ ~~_[a]_~~ _[T]_ is an example of a counterfactual based on a _deterministic static interven-_
_tion_, in which a fixed treatment is applied with probability one and regardless of covariate
information (e.g., _A_ _T_ = ~~_a_~~ _T_ is applied uniformly across units, regardless of past histories
**H** _t_ ). Deterministic static interventions are the kinds of interventions most commonly considered in practice; examples include the average effect of a point exposure E( _Y_ [1] _−_ _Y_ [0] ),
and standard marginal structural model and structural nested model parameters E( _Y_ ~~_[a]_~~ _[T]_ )
and E( _Y_ ~~_[a]_~~ _[t]_ _[,]_ [0] _−_ _Y_ ~~_[a]_~~ _[t][−]_ [1] _[,]_ [0] _|_ **H** _t_ _, A_ _t_ ), respectively (Robins 2000; Robins _et al._ 2000).
Alternatively, in _deterministic dynamic interventions_ (Murphy _et al._ 2001; Robins 1986)
treatment at time _t_ is assigned according to a fixed rule _d_ _t_ : _H_ _t_ _�→A_ that depends on past
history. Characterizing and estimating the optimal such rule is a major goal in the optimal
dynamic treatment regime literature (Murphy 2003; Robins 2004). The potential outcome
under a sequence of hypothetical rules **d** = _d_ _T_ = ( _d_ 1 _, ..., d_ _T_ ) can be expressed as _Y_ **[d]**, where
the dependence of the rules on the histories **H** _t_ is suppressed for notational simplicity, and
**d** is lower-case since the rule is non-random (given the histories). A simple example is the
rule _d_ _t_ = 1( _V_ _t_ _≥_ _c_ _t_ ) that assigns treatment if a variable _V_ _t_ _⊂_ **X** _t_ passes some threshold
_c_ _t_ _∈_ R, with corresponding mean outcome E( _Y_ [(] _[d]_ [1] _[,...,d]_ _[T]_ [ )] ).
In this paper we propose a new form of _stochastic dynamic intervention_, which is an
intervention where treatment at each time is randomly assigned based on a conditional
distribution _q_ _t_ ( _a_ _t_ _|_ **h** _t_ ). Stochastic interventions can thus be viewed as random choices
among deterministic rules. These interventions have not been studied as extensively as
other types, with important exceptions listed in the Introduction (see for example D´ıaz &
van der Laan (2012), Haneuse & Rotnitzky (2013), and Young _et al._ (2014) for review).
We express the potential outcome under a stochastic intervention as _Y_ **[Q]**, where **Q** =
( _Q_ 1 _, ..., Q_ _T_ ) represents draws from the conditional distributions _q_ _t_, and is upper-case since
the intervention is stochastic. A simple stochastic intervention related to the previous rule
_d_ _t_ = 1( _V_ _t_ _≥_ _c_ _t_ ) would be _Q_ _t_ = 1( _V_ _t_ _≥_ _C_ _t_ ) where _C_ _t_ _∼_ _N_ (0 _,_ 1) is now a random threshold.


2


### **3 Incremental Intervention Theory**

In this section we first describe a new class of stochastic dynamic intervention, which we call
incremental propensity score interventions, and give some motivation and examples. We go
on to show that these interventions are nonparametrically identified without requiring any
positivity restrictions on the propensity scores (e.g., the propensity scores do not need to be
bounded away from zero and one). Then we describe the efficiency theory for estimating
mean outcomes under these interventions, based on a new result for general stochastic
interventions that depend on the observational treatment distribution.

#### **3.1 Proposed Interventions**


In this paper we propose _incremental propensity score interventions_ that replace the observational treatment process (i.e., propensity score) _π_ _t_ ( **h** _t_ ) = P( _A_ _t_ = 1 _|_ **H** _t_ = **h** _t_ ) with
a shifted version, based on multiplying the odds of receiving treatment. Specifically, our
proposed intervention replaces the observational propensity score _π_ _t_ with the distribution
defined by

_δπ_ _t_ ( **h** _t_ )
_q_ _t_ ( **h** _t_ ; _δ, π_ _t_ ) = (1)
_δπ_ _t_ ( **h** _t_ ) + 1 _−_ _π_ _t_ ( **h** _t_ ) [for 0] _[ < δ <][ ∞][.]_


The increment parameter _δ ∈_ (0 _, ∞_ ) is user-specified, and dictates the extent to which
the propensity scores are fluctuated from their actual observational values. In practice
we recommend considering a range of _δ_ values, depending on the scientific question at
hand. Although the intervention distribution _q_ _t_ depends on both the increment _δ_ and the
observational propensity _π_ _t_, we often drop this dependence and write _q_ _t_ ( **h** _t_ ; _δ, π_ _t_ ) = _q_ _t_ ( **h** _t_ ) or
just _q_ _t_ to ease notation. As mentioned earlier, D´ıaz & van der Laan (2013, 2012), Haneuse
& Rotnitzky (2013), Moore _et al._ (2012), and Young _et al._ (2014) have considered other
different interventions that depend on the observational treatment process.

Our choice of _q_ _t_ in (1) is motivated by both interpretability and mathematical convenience. In particular it yields




_[q]_ _[t]_ [(] **[h]** _[t]_ [)] _[/][{]_ [1] _[ −]_ _[q]_ _[t]_ [(] **[h]** _[t]_ [)] _[}]_ [odds] _[q]_ [(] _[A]_ _[t]_ [ = 1] _[|]_ **[ H]** _[t]_ [ =] **[ h]** _[t]_ [)]

_π_ _t_ ( **h** _t_ ) _/{_ 1 _−_ _π_ _t_ ( **h** _t_ ) _}_ [=] odds _π_ ( _A_ _t_ = 1 _|_ **H** _t_ = **h** _t_ )



_δ_ = _[q]_ _[t]_ [(] **[h]** _[t]_ [)] _[/][{]_ [1] _[ −]_ _[q]_ _[t]_ [(] **[h]** _[t]_ [)] _[}]_



odds _π_ ( _A_ _t_ = 1 _|_ **H** _t_ = **h** _t_ )



whenever 0 _< π_ _t_ _<_ 1, so the increment _δ_ is just an odds ratio, indicating how the intervention changes the odds of receiving treatment. As with usual odds ratios, if _δ >_ 1 then the
intervention increases the odds of receiving treatment, and if _δ <_ 1 then the intervention
decreases these odds (if _δ_ = 1 then _q_ _t_ = _π_ _t_ so the treatment process is left unchanged). For
example, an intervention with _δ_ = 1 _._ 5 would increase the odds of receiving treatment by
50% for each patient with 0 _< π_ _t_ _<_ 1: a patient with an actual 25% chance of receiving
treatment (1/3 odds) would instead have a 33% chance (1/2 odds) under the intervention.

In addition to other kinds of shifts (e.g., risk ratios) in future work we will consider interventions for which _δ_ = _δ_ _t_ ( **h** _t_ ) can depend on time and covariate history. However, even
with _δ_ fixed, incremental interventions are still dynamic, since the conditional distribution
_q_ _t_ depends on the covariate history. In other words, these interventions are personalized to
patient characteristics through the propensity score. For example, under an intervention


3


with _δ_ = 1 _._ 5, a patient with a 50% chance of receiving treatment observationally would
instead have a 60% chance under the intervention, while a patient whose chances were 5%
would only see an increase to 7.3% (i.e., multiplying the odds by a fixed factor yields different shifts in the probabilities). Contrast this with a usual static intervention, which flatly
assigns all patients a particular sequence ~~_a_~~ _T_ (or a random choice among such sequences)
regardless of propensity score. Figure 1 illustrates incremental interventions with data on
_n_ = 20 simulated observations in a hypothetical study with _T_ = 2 timepoints.


































|Col1|●<br>● ●<br>● ●<br>●●<br>●<br>● ●●<br>●<br>●<br>●<br>● ●<br>●<br>●<br>●<br>●<br>δ = 1<br>● δ = 0.5<br>δ < 1|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
||||||||
||||||||
||||||||
||||||||
|||||●<br>δ = 1<br>δ = 0.5<br>δ < 1|●<br>δ = 1<br>δ = 0.5<br>δ < 1|●<br>δ = 1<br>δ = 0.5<br>δ < 1|
||||||||
||||||||


|Col1|●<br>● ● ● ● ●● ●●● ●<br>●<br>● ● ● ●<br>●<br>●<br>●<br>●<br>δ = 1<br>● δ = 1.5<br>δ > 1|Col3|Col4|Col5|Col6|Col7|Col8|Col9|
|---|---|---|---|---|---|---|---|---|
||||||||||
||||||||||
||||||||||
||||||||||
|||||||●<br>δ = 1<br>δ = 1.5<br>δ > 1|●<br>δ = 1<br>δ = 1.5<br>δ > 1|●<br>δ = 1<br>δ = 1.5<br>δ > 1|
||||||||||
||||||||||



0.0 0.2 0.4 0.6 0.8 1.0


Propensity score at t = 1



0.0 0.2 0.4 0.6 0.8 1.0


Propensity score at t = 1



**Figure 1:** Observational propensity scores for _n_ = 20 simulated units in a study with
_T_ = 2 timepoints, and their values under incremental interventions based on different _δ_
values ( _δ ≤_ 1 in the left plot, _δ ≥_ 1 in the right).


Figure 1 also helps illustrate why incremental interventions require weak identifying
assumptions (to be discussed shortly), and can be more likely to occur in practice, compared
to other kinds of interventions. Although usual static interventions (e.g., setting _A_ = 1)
might require forcing treatment on someone with only a 1% chance of receiving it in the
real world, the proposed incremental intervention only requires that the propensity score be
slightly shifted (e.g., from 1% to 1.5% when _δ_ = 1 _._ 5). In settings where treatment changes
occur more gradually (e.g., when physicians slightly reduce treatment intensity, or judges
become slightly more lenient), incremental interventions might be similar to treatment
changes that could occur naturally in practice. Even if not especially likely to occur,
incremental interventions still might be more realistic than standard static interventions
since they are “closer” to the observational treatment distribution. Of course, incremental
interventions can still be useful analysis tools even if not necessarily mimicking realistic
treatment changes, as discussed in more detail in the next subsection.

One trade-off between incremental and more standard interventions is that, by virtue
of their dependence on the observational treatment process, incremental interventions will
often play a more descriptive rather than prescriptive role. Incremental interventions allow
one to describe how outcomes would vary with gradual changes in treatment intensity; but
they are typically less useful for making specific recommendations about optimal treatment.


4


Nonetheless incremental interventions do generalize common static and dynamic interventions (both deterministic and stochastic), since they can recover these interventions
with particular choices of _δ_ _t_ ( **h** _t_ ). For example, if positivity holds then taking the values
_δ_ = _∞_ and _δ_ = 0 recovers the usual static interventions, yielding potential outcomes _Y_ **[1]**

and _Y_ **[0]** under exposures _A_ _T_ = (1 _, ...,_ 1) and _A_ _T_ = (0 _, ...,_ 0), respectively. Thus incremental
interventions can also be used for a sensitivity analysis of the positivity assumption. If
positivity is violated, then _q_ _t_ _→_ 1( _π_ _t_ _>_ 0) for _δ →∞_, and _q_ _t_ _→_ 1( _π_ _t_ = 1) for _δ →_ 0; these
are the “realistic individualized treatment rules” proposed by Moore _et al._ (2012) and van
der Laan & Petersen (2007), which are dynamic but deterministic. Finally, incremental
interventions can recover general stochastic dynamic interventions (where _q_ _t_ _[∗]_ [replaces the]
propensity score _π_ _t_ ) by taking _δ_ _t_ = _{q_ _t_ _[∗]_ _[/]_ [(1] _[ −]_ _[q]_ _t_ _[∗]_ [)] _[}][/][{][π]_ _[t]_ _[/]_ [(1] _[ −]_ _[π]_ _[t]_ [)] _[}]_ [ for some arbitrary] _[ q]_ _t_ _[∗]_ [,]
whenever defined.

#### **3.2 Identification**


In the previous section we described incremental propensity score interventions, which are
based on shifting the propensity scores _π_ _t_ by multiplying the odds of receiving treatment
by _δ_ . We will now give assumptions that allow for identification of the entire marginal
distribution of the resulting potential outcomes _Y_ **[Q]** [(] _[δ]_ [)], although for simplicity we focus on
estimating just the mean of this distribution.
Importantly, identification of incremental intervention effects requires no conditions on
the propensity scores _π_ _t_, since propensity scores that equal zero or one are not shifted. This
is different from more common interventions that require propensity scores to be bounded
or otherwise restricted in some way. Specifically we only require the following consistency
and exchangeability assumptions.


_Assumption_ 1 (Consistency) _. Y_ = _Y_ ~~_[a]_~~ _[T]_ if _A_ _T_ = ~~_a_~~ _T_ .


_Assumption_ 2 (Exchangeability) _. A_ _t_ _⊥⊥_ _Y_ ~~_[a]_~~ _[T]_ _|_ **H** _t_ .


Consistency means observed outcomes equal corresponding potential outcomes under
the observed treatment sequence; it would be violated for example in network settings
with interference, where outcomes can be affected by other units’ treatment assignment.
Exchangeability means treatment assignment is essentially randomized within covariate
strata; it can hold by design in a trial, but in observational studies it requires sufficiently
many relevant adjustment covariates to be collected. Importantly, no conditions are needed
on the propensity score, since fluctuations based on _q_ _t_ in (1) will leave the propensity score
unchanged if it is zero or one. To the best of our knowledge, the only other work that
has discussed removing positivity conditions entirely is Moore _et al._ (2012) and van der
Laan & Petersen (2007); however, they utilize different (deterministic) interventions and
consider parametric effect models. General interventions could be modified to similarly
avoid positivity, by redefining them to not affect subjects with extreme propensity scores.
Two benefits of incremental interventions are (i) avoiding positivity occurs naturally and
smoothly via the definition of _q_ _t_, rather than an inserted indicator; and (ii) as discussed
shortly, effects under a wide range of treatment intensities can be summarized with a single
curve rather than many regime-specific parameters.


5


The next theorem shows that the mean counterfactual outcome _ψ_ ( _δ_ ) = E( _Y_ **[Q]** [(] _[δ]_ [)] ) under
the incremental intervention is identified and can be expressed uniquely in terms of the
observed data distribution P.


**Theorem 1.** _Under Assumptions 1–2, and if δ ∈D_ = [ _δ_ _ℓ_ _, δ_ _u_ ] _for_ 0 _< δ_ _ℓ_ _≤_ _δ_ _u_ _< ∞, the_
_incremental effect ψ_ ( _δ_ ) = E( _Y_ **[Q]** [(] _[δ]_ [)] ) _equals_



_T_
�


_t_ =1



_a_ _t_ _δπ_ _t_ ( **h** _t_ ) + (1 _−_ _a_ _t_ ) _{_ 1 _−_ _π_ _t_ ( **h** _t_ ) _}_

_d_ P( **x** _t_ _|_ **h** _t−_ 1 _, a_ _t−_ 1 )
_δπ_ _t_ ( **h** _t_ ) + 1 _−_ _π_ _t_ ( **h** _t_ )



_ψ_ ( _δ_ ) = �

~~_a_~~ _T_ _∈A_ _[T]_



_µ_ ( **h** _T_ _, a_ _T_ )

�

_X_



_where X_ = _X_ 1 _× · · · × X_ _T_ _and µ_ ( **h** _T_ _, a_ _T_ ) = E( _Y |_ **H** _T_ = **h** _T_ _, A_ _T_ = _a_ _T_ ) _._


Proofs of all theorems are given in the Appendix. Theorem 1 follows from Robins’
g-formula (Robins 1986), replacing the general treatment process under intervention with
the proposed incremental intervention _q_ _t_ indexed by _δ_ . The next corollary shows how the
expression for _ψ_ ( _δ_ ) simplifies in point exposure studies.


**Corollary 1.** _When T_ = 1 _the identifying expression for ψ_ ( _δ_ ) _simplifies to_



_δπ_ ( **X** ) _µ_ ( **X** _,_ 1) + _{_ 1 _−_ _π_ ( **X** ) _}µ_ ( **X** _,_ 0)
_ψ_ ( _δ_ ) = E
� _δπ_ ( **X** ) + _{_ 1 _−_ _π_ ( **X** ) _}_


_with µ_ ( **x** _, a_ ) = E( _Y |_ **X** = **x** _, A_ = _a_ ) _._



�



This corollary shows that, when _T_ = 1, the incremental effect _ψ_ ( _δ_ ) is a weighted average
of the regression functions _µ_ ( **x** _,_ 1) and _µ_ ( **x** _,_ 0), where the weight on _µ_ ( **x** _,_ 1) is given by the
fluctuated intervention propensity score _q_ ( **x** ) = _δπ_ ( **x** ) _/{δπ_ ( **x** ) + 1 _−_ _π_ ( **x** ) _}_ (and the weight
on _µ_ ( **x** _,_ 0) is 1 _−_ _q_ ( **x** )). This weight tends to zero as _δ →_ 0 (whenever _π_ ( **x** ) _<_ 1) and tends
to one for _δ →∞_ (whenever _π_ ( **x** ) _>_ 0), showing again that _δ_ controls how far away the
intervention is from the observational treatment process. Incremental interventions can
range from assigning no one to everyone treatment, but also include an infinite middle
ground. Note that we can also write _ψ_ ( _δ_ ) = E _{µ_ ( **X** _, A_ _[∗]_ ) _}_ where _A_ _[∗]_ is a simulated version
of treatment under the incremental intervention, with ( _A_ _[∗]_ _|_ **X** = **x** ) _∼_ Bernoulli _{q_ ( **x** ) _}_ .

Beyond the fact that identifying incremental effects does not require positivity conditions, targeting _ψ_ ( _δ_ ) has another crucial advantage: it is always a one-dimensional curve,
regardless of the number of timepoints _T_, and even though it characterizes infinitely many
interventions nonparametrically. In contrast, for more traditional causal effects, there is a
distinct tension between the number of hypothetical interventions studied and the complexity of the effect. For example one could consider the mean outcome E( _Y_ ~~_[a]_~~ _[T]_ ) under all 2 _[T]_

deterministic interventions ~~_a_~~ _T_ _∈{_ 0 _,_ 1 _}_ _[T]_, but this requires exponentially many parameters
without further assumptions. One could impose smoothness across the 2 _[T]_ interventions
to reduce the parameter space, but this will yield bias if the smoothness assumptions are
incorrect. Conversely, describing the mean outcome under a small number of interventions
such as ~~_a_~~ _T_ = **0** and ~~_a_~~ _T_ = **1** (i.e., never treated and always treated) requires only a few
parameters, but gives a very limited picture of how changing treatment affects outcomes.
In contrast, incremental interventions allow exploration of infinitely many interventions
(one for each _δ ∈D_ ), without any parametric assumptions, regardless of how large _T_ is,
and still only yield a single curve _ψ_ : _D �→_ R that can be easily visualized with a plot.


6


#### **3.3 Efficiency Theory**

So far we have introduced incremental propensity score interventions, and showed that
resulting effects can be identified without requiring positivity assumptions. Now we will
develop general efficiency theory for the incremental effect _ψ_ ( _δ_ ) = E( _Y_ **[Q]** [(] _[δ]_ [)] ).
We refer elsewhere (Bickel _et al._ 1993; Kennedy 2016; Tsiatis 2006; van der Laan &
Robins 2003; van der Vaart 2002) for more detailed information about nonparametric efficiency theory, and so give only a brief review here. A fundamental goal is characterizing
so-called influence functions, and in particular finding the efficient influence function. These
tasks are essential for a number of reasons. Perhaps most importantly, influence functions
can be used to construct estimators with very favorable properties, such as double robustness or general second-order bias (called Neyman orthogonality by Chernozhukov _et al._
(2016)). Estimators with these properties can attain fast parametric convergence rates,
even in nonparametric settings where nuisance functions are estimated at slower rates via
flexible machine learning. The efficient influence function (the only influence function in
fully nonparametric models) is particularly important since its variance equals the efficiency bound, thus providing an important benchmark and allowing for the construction of
optimal estimators. Influence functions are also critical for understanding the asymptotics
of corresponding estimators, since by definition any regular asymptotically linear estimator can be expressed as the empirical average of an influence function plus a negligible
_o_ _p_ (1 _/_ _[√]_ ~~_n_~~ ~~)~~ error term.
Mathematically, influence functions are essentially derivatives. More specifically, viewed
as elements of the Hilbert space of mean-zero finite-variance functions, influence functions
are those elements whose covariance with parametric submodel scores equals a pathwise
derivative of the target parameter. Influence functions also correspond to the derivative in a
Von Mises expansion of the target parameter (a distributional analog of a Taylor expansion),
and in nonparametric models with discrete support they are a Gateaux derivative of the
parameter in the direction of a point mass contamination.
The result of the next theorem is an expression for the efficient influence function for
the incremental effect _ψ_ ( _δ_ ) under a nonparametric model, which allows the data-generating
process P to be infinite-dimensional. This efficient influence function can be used to characterize the efficiency bound for estimating _ψ_ ( _δ_ ), and we will see how this bound changes in
randomized trial settings where the propensity scores are known. Then in the next section
the efficient influence function will be used to construct estimators, including optimally
efficient estimators with the second-order bias property discussed earlier.


**Theorem 2.** _The efficient influence function for ψ_ ( _δ_ ) _under a nonparametric model (with_
_unknown propensity scores) is given by_



_T_
�


_t_ =1



�



_A_ _t_ _{_ 1 _−_ _π_ _t_ ( **H** _t_ ) _} −_ (1 _−_ _A_ _t_ ) _δπ_ _t_ ( **H** _t_ )
� _δ/_ (1 _−_ _δ_ )



_δπ_ _t_ ( **H** _t_ ) _m_ _t_ ( **H** _t_ _,_ 1) + _{_ 1 _−_ _π_ _t_ ( **H** _t_ ) _}m_ _t_ ( **H** _t_ _,_ 0)
�� _δπ_ _t_ ( **H** _t_ ) + 1 _−_ _π_ _t_ ( **H** _t_ )



�



_T_
�


_t_ =1



( _δA_ _t_ + 1 _−_ _A_ _t_ ) _Y_
_δπ_ _t_ ( **H** _t_ ) + 1 _−_ _π_ _t_ ( **H** _t_ ) _[−]_ _[ψ]_ [(] _[δ]_ [)]



_×_



_t_
�
� _s_ =1



( _δA_ _s_ + 1 _−_ _A_ _s_ )
_δπ_ _s_ ( **H** _s_ ) + 1 _−_ _π_ _s_ ( **H** _s_ )



+



7


_where for t_ = 0 _, ..., T −_ 1 _we define_



_m_ _t_ ( **h** _t_ _, a_ _t_ ) = _µ_ ( **h** _T_ _, a_ _T_ )
� _R_ _t_



_T_
�


_s_ = _t_ +1



_a_ _s_ _δπ_ _s_ ( **h** _s_ ) + (1 _−_ _a_ _s_ ) _{_ 1 _−_ _π_ _s_ ( **h** _s_ ) _}_

_d_ P( **x** _s_ _|_ **h** _s−_ 1 _, a_ _s−_ 1 )
_δπ_ _s_ ( **h** _s_ ) + 1 _−_ _π_ _s_ ( **h** _s_ )



_with R_ _t_ = ( _H_ _T_ _× A_ _T_ ) _\ H_ _t_ _, and for t_ = _T we let m_ _T_ ( **h** _T_ _, a_ _T_ ) = _µ_ ( **h** _T_ _, a_ _T_ ) _._


We give a proof of Theorem 2 in Section 8.2 of the Appendix, by way of deriving the
efficient influence function for general stochastic interventions with treatment distributions
that depend on the observational propensity scores. To the best of our knowledge this
result has not yet appeared in the literature, and will be useful for general stochastic
interventions beyond those with the incremental form proposed here, regardless of whether
they depend on the observational treatment process or not. Our result recovers previously
proposed influence functions for other stochastic intervention effects in the _T_ = 1 setting
as special cases (D´ıaz & van der Laan 2012; Haneuse & Rotnitzky 2013), and could be
used to generalize this work to the multiple timepoint setting. Further, our result can
also be used to construct the efficient influence function and corresponding estimator for
other stochastic intervention effects, for which there are currently only likelihood-based and
weighting estimators available (Moore _et al._ 2012; Young _et al._ 2014).

The structure of the efficient influence function in Theorem 2 is somewhat similar to that
of more standard effect parameters, in the sense that it consists of an inverse-probabilityweighted term (the rightmost product term in the second line) as well as an augmentation
term. However the particular form of the weighted and augmentation terms are quite
different from those that appear in more common causal and missing data problems. We
discuss the weighted term in more detail in Section 4.1, when we introduce an inverseprobability-weighted estimator for _ψ_ ( _δ_ ). The augmentation term involves the functions _m_ _t_,
which can be viewed as marginalized versions of the full regression function _µ_ ( **h** _t_ _, a_ _t_ ) that
conditions on all of the past (with smaller values of _t_ coinciding with more marginalization).
Note that for notational simplicity we drop the dependence of _m_ _t_ on _δ_ and ( _π_ _t_ +1 _, ..., π_ _T_ ),
as well as on the conditional densities of the covariates ( **X** _t_ +1 _, ...,_ **X** _T_ ). Importantly, the
pseudo-regression functions _m_ _t_ also have a recursive sequential regression formulation, as
displayed in the subsequent remark.


_Remark_ 2 _._ The functions _m_ _t_ can be equivalently expressed recursively as



_δπ_ _t_ ( **H** _t_ ) _m_ _t_ ( **H** _t_ _,_ 1) + _{_ 1 _−_ _π_ _t_ ( **H** _t_ ) _}m_ _t_ ( **H** _t_ _,_ 0)
_m_ _t−_ 1 ( **H** _t−_ 1 _, A_ _t−_ 1 ) = E
� _δπ_ _t_ ( **H** _t_ ) + 1 _−_ _π_ _t_ ( **H** _t_ )


for _t_ = 1 _, ..., T_ and _m_ _T_ ( **h** _T_ _, a_ _T_ ) = _µ_ ( **h** _T_ _, a_ _T_ ) as before.



**H** _t−_ 1 _, A_ _t−_ 1
���



�



Viewing the _m_ _t_ functions in the above sequential regression form is very practically
useful for the purposes of estimation. Specifically it shows how to bypass conditional
density estimation, and instead construct estimates ˆ _m_ _t_ using regression methods that are
more commonly found in statistical software.
It is also important to note that the pseudo-regressions _m_ _t_ depend on the observational
treatment process; this is not the case for analogous influence function terms for more
common parameters like E( _Y_ ~~_[a]_~~ _[T]_ ). This is due to the fact that the functional _ψ_ ( _δ_ ) itself


8


depends on the observational treatment process, which means for example that double
robustness is not possible (though second-order bias still is) and that the efficiency bound
is different when the propensity scores are known versus unknown. The issue of double
robustness is discussed in more detail in Section 4.3. In Lemmas 2 and 4 in the Appendix
we give the efficient influence function when the propensity scores are known, as well as a
specific expression for the contribution that comes from the scores being unknown, both
for general (possibly non-incremental) stochastic interventions.
In the next corollary we give the efficient influence function for the incremental effect
in a single timepoint study, which has a simpler and more intuitive form.


**Corollary 2.** _When T_ = 1 _the efficient influence function for ψ_ ( _δ_ ) _simplifies to_


_δπ_ ( **X** ) _φ_ 1 ( **Z** ) + _{_ 1 _−_ _π_ ( **X** ) _}φ_ 0 ( **Z** ) _δγ_ ( **X** ) _{A −_ _π_ ( **X** ) _}_

+

_[−]_ _[ψ]_ [(] _[δ]_ [)]
_δπ_ ( **X** ) + _{_ 1 _−_ _π_ ( **X** ) _}_ _{δπ_ ( **X** ) + 1 _−_ _π_ ( **X** ) _}_ [2]


_where γ_ ( **x** ) = _µ_ ( **x** _,_ 1) _−_ _µ_ ( **x** _,_ 0) _and_


_φ_ _a_ ( **Z** ) = [1][(] _[A]_ [ =] _[ a]_ [)]

_π_ ( _a |_ **X** ) _[{][Y][ −]_ _[µ]_ [(] **[X]** _[, a]_ [)] _[}]_ [ +] _[ µ]_ [(] **[X]** _[, a]_ [)]


_is the uncentered efficient influence function for the parameter_ E _{φ_ _a_ ( **Z** ) _}_ = E _{µ_ ( **X** _, a_ ) _}._


The efficient influence function in the _T_ = 1 case is therefore a simple weighted average
of the influence functions for E( _Y_ [1] ) and E( _Y_ [0] ), plus a contribution that comes from the
fact that the propensity score is unknown and must be estimated. If the propensity scores
were known, the efficient influence function would just be the first weighted average term
in Corollary 2. As will be discussed in more detail in the next section, estimating the
influence function in the _T_ = 1 case is straightforward as it only depends on the regression
function _µ_ and propensity score _π_ (rather than the sequential psuedo-regression functions
_m_ _t_ that appear in the longitudinal setting).

### **4 Estimation & Inference**


In this section we develop estimators for the proposed incremental effect _ψ_ ( _δ_ ). We focus
our analysis on flexible sample-splitting estimators that allow arbitrarily complex nuisance
estimation, e.g., via high-dimensional regression and machine learning methods; however
we also discuss simpler estimators that rely on empirical process conditions to justify fullsample nuisance estimation. In particular we show that there exists an inverse-probabilityweighted estimator of the incremental effect that is especially easy to compute. We go
on to describe the asymptotic behavior of our proposed estimators, both from a pointwise
perspective and uniformly across a continuum of increment parameter _δ_ values. Finally we
propose a computationally efficient multiplier-bootstrap approach for constructing uniform
confidence bands across _δ_, and use it to develop a novel test of no treatment effect.


9


#### **4.1 Simple Estimators**

We first describe various simple estimators of the incremental effect, which provide some
intuition for the main estimator we propose in the next section. The simple inverseprobability-weighted estimator discussed here might be preferred if the propensity scores
can be modeled well (e.g., in a randomized trial) and computation comes at a high cost.
Let _ϕ_ ( **Z** ; _**η**_ _, δ_ ) denote the (uncentered) efficient influence function from Theorem 2,
which is a function of the observations **Z** and the nuisance functions


_**η**_ = ( _**π**_ _,_ **m** ) = ( _π_ 1 _, ..., π_ _T_ _, m_ 1 _, ..., m_ _T_ ) _._


By uncentered we mean that _ϕ_ ( **Z** ; _**η**_ _, δ_ ) equals the quantity displayed in Theorem 2 plus
the parameter _ψ_ ( _δ_ ), so that E _{ϕ_ ( **Z** ; _**η**_ _, δ_ ) _}_ = _ψ_ ( _δ_ ) by construction.
If one is willing to rely on appropriate empirical process conditions (e.g., Donsker-type
or low entropy conditions, as discussed by van der Vaart & Wellner (1996), van der Vaart
(2000), and others) then a natural estimator would be given by the solution to the efficient
influence function estimating equation, i.e., the Z-estimator


ˆ
_ψ_ _[∗]_ ( _δ_ ) = P _n_ _{ϕ_ ( **Z** ; ˆ _**η**_ _, δ_ ) _}_



where ˆ _**η**_ are some initial estimators of the nuisance functions, and P _n_ denotes the empirical
measure so that sample averages can be written as [1] � _[f]_ [(] **[Z]** _[i]_ [) =][ P] _[n]_ _[{][f]_ [(] **[Z]** [)] _[}]_ [ =] � _f_ ( **z** ) _d_ P _n_ ( **z** ).



measure so that sample averages can be written as _n_ [1] � _i_ _[f]_ [(] **[Z]** _[i]_ [) =][ P] _[n]_ _[{][f]_ [(] **[Z]** [)] _[}]_ [ =] � _f_ ( **z** ) _d_ P _n_ ( **z** ).

An algorithm describing how to compute the estimator _ψ_ [ˆ] _[∗]_ ( _δ_ ) is given in Section 8.3 of
the Appendix. As a special case, if the propensity scores _π_ _t_ can be correctly modeled
parametrically (e.g., when they are known as in a randomized trial) then one could use the
simple inverse-probability-weighted estimator given by



_n_ [1] �



�



ˆ
_ψ_ _ipw_ _[∗]_ [(] _[δ]_ [) =][ P] _[n]_



_T_
�
� _t_ =1



( _δA_ _t_ + 1 _−_ _A_ _t_ ) _Y_
_δπ_ ˆ _t_ ( **H** _t_ ) + 1 _−_ _π_ ˆ _t_ ( **H** _t_ )



_._



This estimator can be computed very quickly, as it only requires fitting a single pooled
regression to estimate _π_ _t_ and then taking a weighted average. However it has disadvantages,
as will be discussed shortly. Also note that it is a special case of _ψ_ [ˆ] _[∗]_ ( _δ_ ) that sets ˆ _m_ _t_ = 0.
It is instructive to compare the inverse-probability-weighted estimator above to that
for a usual deterministic static intervention effect like E( _Y_ ~~_[a]_~~ _[T]_ ). For example, the inverseprobability-weighted estimator of the quantity E( _Y_ **[1]** ) weights each always-treated unit
by the (inverse) product of propensity scores [�] _t_ _[π]_ [ˆ] _[t]_ [, and otherwise assigns zero weight. In]

contrast, when _δ >_ 1 the estimator _ψ_ [ˆ] _ipw_ _[∗]_ [(] _[δ]_ [) weights each treated time by the (inverse of the)]

ˆ
propensity score plus some fractional contribution of its complement, i.e., ˆ _π_ _t_ + (1 _−_ _π_ _t_ ) _/δ_,
where the size of the contribution decreases with _δ_ ; untreated times are weighted by this
same amount, except the entire weight is further downweighted by a factor of _δ_ . Therefore
when _δ_ is very large, the two inverse-probability-weighted estimators coincide. However,
for cases when _δ_ is not very large, this also indicates why the estimator _ψ_ [ˆ] _ipw_ _[∗]_ [(] _[δ]_ [) is immune]
to extreme weights: even if ˆ _π_ _t_ is very small, there will still be a contribution to the weight
that moves it away from zero.


10


#### **4.2 Proposed Estimator**

Although the estimators presented in the previous section are relatively simple, they have
some disadvantages. First, the inverse-probability-weighted estimator _ψ_ [ˆ] _ipw_ _[∗]_ [(] _[δ]_ [) will in gen-]
eral not be _[√]_ ~~_n_~~ ~~-~~ consistent unless all the propensity scores are estimated with correctly
specified parametric models; this is typically an unreasonable assumption outside of randomized trials where propensity scores are known. In point exposure studies with a single
timepoint, (saturated) parametric models might be used if the adjustment covariates are
low-dimensional. However, in studies with more than just a few timepoints, the histories
**H** _t_ can easily be high-dimensional even if the covariates **X** _t_ are low-dimensional, making
parametric modeling assumptions less tenable even in the low-dimensional **X** _t_ case.
In contrast, the more general Z-estimator _ψ_ [ˆ] _[∗]_ ( _δ_ ) can converge at fast parametric _[√]_ ~~_n_~~
rates (and attain the efficiency bound from Section 3.3), even when the propensity scores
_π_ _t_ and pseudo-outcome regressions _m_ _t_ are modeled flexibly and estimated at rates slower
than _[√]_ ~~_n_~~ ~~,~~ as long as these nuisance functions are estimated consistently at rates faster
than _n_ [1] _[/]_ [4] . Lowering the bar from _[√]_ ~~_n_~~ to _n_ [1] _[/]_ [4] for the nuisance estimator convergence rate
allows much more flexible nonparametric methods to be employed; for example these rates
are attainable under smoothness, sparsity, or other nonparametric structural constraints.
However, as mentioned earlier, these Z-estimator properties require some empirical process
conditions that restrict the flexibility and complexity of the nuisance estimators. This is
essentially because _ψ_ [ˆ] _[∗]_ ( _δ_ ) uses the sample twice, once for estimating the nuisance functions
_**η**_ and again for evaluating the influence function _ϕ_ . Without restricting the entropy of
the nuisance estimators, using the full sample in this way can result in overfitting and
intractable asymptotics. Unfortunately, the required empirical process conditions may not
be satisfied by many modern regression methods, such as random forests, boosting, deep
learning, or complicated ensembles.
In order to accommodate the added complexity of these modern machine learning tools,
we use sample splitting (Chernozhukov _et al._ 2016; Zheng & van der Laan 2010). This avoids
the problematic “double” use of the sample and, as will be seen in the next section, yields
asymptotically normal and efficient estimators without any restrictions on the complexity
of the nuisance estimators (however, _n_ [1] _[/]_ [4] -type rate conditions are still required).
Therefore we randomly split the observations ( **Z** 1 _, ...,_ **Z** _n_ ) into _K_ disjoint groups, using
a random variable _S_ drawn independently of the data, where _S_ _i_ _∈{_ 1 _, ..., K}_ denotes the
group membership for unit _i_ . Then our proposed estimator is given by



ˆ
_ψ_ ( _δ_ ) = [1]

_K_



_K_
� P _[k]_ _n_ _[{][ϕ]_ [(] **[Z]** [; ˆ] _**[η]**_ - _k_ _[, δ]_ [)] _[}]_ [ =][ P] _[n]_ _[{][ϕ]_ [(] **[Z]** [; ˆ] _**[η]**_ - _S_ _[, δ]_ [)] _[}]_


_k_ =1



where we let P _[k]_ _n_ [denote empirical averages only over the set of units] _[ {][i]_ [ :] _[ S]_ _[i]_ [=] _[ k][}]_ [ in group]
_k_ (i.e., P _n_ _{f_ ( **Z** ) _}_ = [�] _i_ _[f]_ [(] **[Z]** _[i]_ [)][1][(] _[S]_ _[i]_ [ =] _[ k]_ [)] _[/]_ [ �] _i_ [1][(] _[S]_ _[i]_ [ =] _[ k]_ [)), and we let ˆ] _**[η]**_ - _k_ [denote the nuisance]



_k_ (i.e., P _n_ _{f_ ( **Z** ) _}_ = [�] _i_ _[f]_ [(] **[Z]** _[i]_ [)][1][(] _[S]_ _[i]_ [ =] _[ k]_ [)] _[/]_ [ �] _i_ [1][(] _[S]_ _[i]_ [ =] _[ k]_ [)), and we let ˆ] _**[η]**_ - _k_ [denote the nuisance]

estimator constructed excluding group _k_, i.e., only using those units _{i_ : _S_ _i_ _̸_ = _k}_ in groups
_K\k_ . It is hoped that ˆ _**η**_ - _k_ is a rate-optimal estimator of the nuisance functions, for example
constructed using kernels, splines, penalized regression, boosting, random forests, etc., or
some ensemble-based combination.



_i_ _[f]_ [(] **[Z]** _[i]_ [)][1][(] _[S]_ _[i]_ [ =] _[ k]_ [)] _[/]_ [ �]



11


An algorithm detailing exactly how to compute the estimator _ψ_ [ˆ] ( _δ_ ) is given as follows.
For reference, the algorithm for the non-sample splitting estimator _ψ_ [ˆ] _[∗]_ ( _δ_ ) is also given in
Section 8.3 of the Appendix and contains the main ideas.


**Algorithm 1.** _For each δ and k, letting_ **D** 0 = _{_ **Z** _i_ : _S_ _i_ _̸_ = _k} and_ **D** 1 = _{_ **Z** _i_ : _S_ _i_ = _k}_
_denote corresponding training and test data, respectively, and_ **D** = **D** 0 _∪_ **D** 1 _:_


_1. Regress A_ _t_ _on_ **H** _t_ _in_ **D** 0 _, obtain predicted values_ ˆ _π_ _t_ ( **H** _t_ ) _for each subject/time in_ **D** _._


_δA_ _t_ +1 _−A_ _t_
_2. Construct time-dependent weights W_ _t_ = _δπ_ ˆ _t_ ( **H** _t_ )+1 _−π_ ˆ _t_ ( **H** _t_ ) _[in]_ **[ D]** [1] _[ for each subject/time.]_


_3. Calculate cumulative product weight_ _W_ [�] _t_ = [�] _[t]_ _s_ =1 _[W]_ _[s]_ _[ in]_ **[ D]** [1] _[ for each subject/time.]_


_4. For each time t_ = _T, T −_ 1 _, ...,_ 1 _(starting with R_ _T_ +1 = _Y ):_


_(a) Regress R_ _t_ +1 _on_ ( **H** _t_ _, A_ _t_ ) _in_ **D** 0 _, obtain predictions_ ˆ _m_ _t_ ( **H** _t_ _,_ 1) _,_ ˆ _m_ _t_ ( **H** _t_ _,_ 0) _in_ **D** _._

_(b) Construct pseudo-outcome R_ _t_ = _[δ][π]_ [ˆ] _[t]_ [(] **[H]** _[t]_ [)][ ˆ] _[m]_ _[t]_ _δ_ [(] _π_ ˆ **[H]** _t_ ( _[t]_ **H** _[,]_ [1][)+] _t_ )+1 _[{]_ [1] _[−]_ _−_ _[π]_ _π_ [ˆ] ˆ _[t]_ _t_ [(] ( **[H]** **H** _[t]_ _t_ [)] ) _[}]_ [ ˆ] _[m]_ _[t]_ [(] **[H]** _[t]_ _[,]_ [0][)] _in_ **D** _._


_5. Compute time-dependent weights V_ _t_ = _[A]_ _[t]_ _[{]_ [1] _[−][π]_ [ˆ] _[t]_ [(] **[H]** _[t]_ _δ/_ [)] _[}]_ (1 _[−]_ _−_ [(][1] _δ_ _[−]_ ) _[A]_ _[t]_ [)] _[δ][π]_ [ˆ] _[t]_ [(] **[H]** _[t]_ [)] _in_ **D** 1 _._


_6. Compute ϕ_ = _W_ [�] _T_ _Y_ + [�] _t_ _W_ [�] _t_ _V_ _t_ _R_ _t_ _in_ **D** 1 _and define_ _ψ_ [ˆ] _k_ ( _δ_ ) _to be its average in_ **D** 1 _._

_Finally, set_ _ψ_ [ˆ] ( _δ_ ) _to be the average of the K estimators_ _ψ_ [ˆ] _k_ ( _δ_ ) _, k_ = 1 _, ..., K._

Importantly, computing _ψ_ [ˆ] ( _δ_ ) only requires estimating regression functions (e.g., using
random forests) and not conditional densities, due to the recursive regression formulation
of the functions _m_ _t_ in Remark 2. Although the process can be somewhat computationally
expensive depending on the number of timepoints _T_, sample size _n_, and grid density for _δ_,
it is easily parallelizable due to the sample splitting. For a single timepoint all estimators
are easy and fast to compute. In Section 8.6 of the Appendix, we provide a user-friendly R
function for general use in cross-sectional or longitudinal studies; the function can also be
found in the `npcausal` R package available at GitHub (github.com/ehkennedy/npcausal).

#### **4.3 Weak Convergence**


In this section we detail the main large-sample property of our proposed estimator, thatˆ
_ψ_ ( _δ_ ) is _[√]_ ~~_n_~~ ~~-~~ consistent and asymptotically normal under weak conditions (mostly only requiring that the nuisance functions are estimated at faster than _n_ [1] _[/]_ [4] rates). This result
holds both pointwise for a given _δ_, and uniformly in the sense that, after scaling and when
viewed as a random function on _D_ = [ _δ_ _ℓ_ _, δ_ _u_ ], the estimator converges in distribution to a
Gaussian process. The latter fact is crucial for developing uniform confidence bands, as
well as the test of no treatment effect we present in the next section. Importantly, the
estimator attains fast _[√]_ ~~_n_~~ rates even under nonparametric assumptions and even though
the target parameter is a curve; this is often not possible (Kennedy _et al._ 2017, 2016).

In what follows we denote the squared _L_ 2 (P) norm by _∥f_ _∥_ [2] = � _f_ ( **z** ) [2] _d_ P( **z** ). When
necessary, we depart slightly from previous sections and index the pseudo-regression functions _m_ _t,δ_ (and their estimators ˆ _m_ _t,δ_ ) by both time _t_ and the increment parameter _δ_ . The
next result lays the foundation for our proposed inferential and testing procedures.


12


**Theorem 3.** _Let_ ˆ _σ_ [2] ( _δ_ ) = P _n_ [ _{ϕ_ ( **Z** ; ˆ _**η**_ _-S_ _, δ_ ) _−_ _ψ_ [ˆ] ( _δ_ ) _}_ [2] ] _denote the estimator of the variance_
_function σ_ [2] ( _δ_ ) = E[ _{ϕ_ ( **Z** ; _**η**_ _, δ_ ) _−_ _ψ_ ( _δ_ ) _}_ [2] ] _. Assume:_


_1. The set D_ = [ _δ_ _ℓ_ _, δ_ _u_ ] _is bounded with_ 0 _< δ_ _ℓ_ _≤_ _δ_ _u_ _< ∞._


_2._ P _{|m_ _t_ ( **H** _t_ _, A_ _t_ ) _| ≤_ _C}_ = P _{|m_ ˆ _t_ ( **H** _t_ _, A_ _t_ ) _| ≤_ _C}_ = 1 _for some C < ∞_ _and all t._


_3._ sup _δ∈D_ _|_ _[σ]_ _σ_ [ˆ][(] ( _[δ]_ _δ_ [)] ) _[−]_ [1] _[|]_ [ =] _[ o]_ [P] [(1)] _[, and][ ∥]_ [sup] _[δ][∈D]_ _[ |][ϕ]_ [(] **[z]** [; ˆ] _**[η]**_ _[, δ]_ [)] _[ −]_ _[ϕ]_ [(] **[z]** [;] _**[ η]**_ _[, δ]_ [)] _[| ∥]_ [=] _[ o]_ [P] [(1)] _[.]_



_4._



ˆ ˆ ˆ
sup _δ∈D_ _∥m_ _t,δ_ _−_ _m_ _t,δ_ _∥_ + _∥π_ _t_ _−_ _π_ _t_ _∥_ _∥π_ _s_ _−_ _π_ _s_ _∥_ = _o_ P (1 _/_ _[√]_ ~~_n_~~ ~~)~~ _for s ≤_ _t ≤_ _T_ _._
� �



_Then_ ˆ
_ψ_ ( _δ_ ) _−_ _ψ_ ( _δ_ )

ˆ ⇝ G( _δ_ )
_σ_ ( _δ_ ) _/_ ~~_[√]_~~ ~~_n_~~


_in ℓ_ _[∞]_ ( _D_ ) _, where_ G( _·_ ) _is a mean-zero Gaussian process with covariance_ E _{_ G( _δ_ 1 )G( _δ_ 2 ) _}_ =
E _{ϕ_ �( **Z** ; _**η**_ _, δ_ 1 ) _ϕ_ �( **Z** ; _**η**_ _, δ_ 2 ) _} and_ � _ϕ_ ( **z** ; _**η**_ _, δ_ ) = _{ϕ_ ( **z** ; _**η**_ _, δ_ ) _−_ _ψ_ ( _δ_ ) _}/σ_ ( _δ_ ) _._


The proof of Theorem 3 is given in Section 8.4 of the Appendix. The logic of the proof is
roughly similar to that used by Belloni _et al._ (2015), but we avoid their restrictions on nuisance function entropy by sample-splitting and arguing conditionally on the training data.
This allows for the use of arbitrarily complex estimators ˆ _**η**_, such as random forests, boosting, etc. We also do not need explicit smoothness assumptions on _ψ_ ( _δ_ ) or _ϕ_ ( **Z** ; _**η**_ _, δ_ ) since
they are necessarily Lipschitz in _δ_ by construction, based on our choice of the incremental
intervention distribution _q_ _t_ .
Assumptions 1–2 of Theorem 3 are mild boundedness conditions on the set _D_ of _δ_ values
and the functions _m_ _t_ and their estimators, respectively. Assumption 2 could be relaxed at
the expense of a less simple proof, for example with bounds on _L_ _p_ norms. Assumption 3 is
a basic and mild consistency assumption, with no requirement on rates of convergence. The
main substantive assumption is Assumption 4, which says the nuisance estimators must be
consistent and converge at a fast enough rate (essentially _n_ [1] _[/]_ [4] in _L_ 2 norm).
Importantly, the rate condition in Assumption 4 can be attained under nonparametric
smoothness, sparsity, or other structural constraints. We are agnostic about how such
rates might be attained since the particular required assumptions are problem-dependent;
in practice we suggest using ensemble learners that can adapt to diverse kinds of structure.
The particular form of the rate requirement indicates that double robustness is not possible,

ˆ ˆ
since we need products of the form _∥π_ _t_ _−_ _π_ _t_ _∥∥π_ _s_ _−_ _π_ _s_ _∥_ to be small, thus requiring consistent
estimation of the propensity scores (albeit only at slower than parametric rates). If the
propensity scores are known as in a randomized trial, then Assumption 4 will necessarily
hold; in this case, the result of the theorem follows with _ϕ_ ( **z** ; _**η**_ _, δ_ ) evaluated at ~~_m_~~ _t_ the limit
of the estimator ˆ _m_ _t_, which may or may not equal the true pseudo-regression _m_ _t_ . If the
propensity scores are estimated with correct parametric models, then Assumption 4 would
only require a (uniformly) consistent estimator of _m_ _t_, without any rate conditions.
Based on the result in Theorem 3, pointwise 95% confidence intervals for _ψ_ ( _δ_ ) can be
constructed as

ˆ
_ψ_ ( _δ_ ) _±_ 1 _._ 96 ˆ _σ_ ( _δ_ ) _/_ _[√]_ _n_


13


where ˆ _σ_ [2] ( _δ_ ) = P _n_ [ _{ϕ_ ( **Z** ; ˆ _**η**_ - _S_ _, δ_ ) _−_ _ψ_ [ˆ] ( _δ_ ) _}_ [2] ] is the variance estimator given in the statement
of the theorem. Uniform inference and testing is discussed in the next section.

#### **4.4 Uniform Inference & Testing No Effect**


In this section we present a multiplier bootstrap approach to obtaining uniform confidence
bands for the incremental effect curve _{ψ_ ( _δ_ ) : _δ ∈D}_, along with a corresponding novel
test of no treatment effect. This test can be useful in general causal inference problems,
even when positivity assumptions are justified and even if incremental effects are not of
particular interest.

ˆ
To construct a (1 _−_ _α_ ) uniform confidence band of the form _ψ_ [ˆ] ( _δ_ ) _±_ _c_ _α_ _σ_ ( _δ_ ) _/_ _[√]_ ~~_n_~~ ~~,~~ as usual
we need to find a critical value _c_ _α_ that satisfies



�����



_≤_ _c_ _α_
�����



�



P



sup
_δ∈D_

�



ˆ
_ψ_ ( _δ_ ) _−_ _ψ_ ( _δ_ )


ˆ
_σ_ ( _δ_ ) _/_ ~~_[√]_~~ ~~_n_~~



= 1 _−_ _α_ + _o_ (1) _,_



since the expression on the left is the probability that the band covers the true incremental
effect curve _ψ_ ( _δ_ ) for all _δ ∈D_ .
Based on the result of Theorem 3, this critical value can be obtained by approximating
the distribution of the supremum of the Gaussian process _{_ G( _δ_ ) : _δ ∈D}_ with covariance
function as given in the statement of the theorem. We use the multiplier bootstrap (Belloni
_et al._ 2015; Gin´e & Zinn 1984; van der Vaart & Wellner 1996) to approximate this distribution. A primary advantage of the multiplier bootstrap is its computational efficiency,
since it does not require refitting the nuisance estimators, which can be expensive when
there are many covariates and/or timepoints.
The idea behind the multiplier bootstrap is to approximate the distribution of the
aforementioned supremum with the supremum of the multiplier process

_√n_ P _n_ _ξ{ϕ_ ( **Z** ; ˆ _**η**_         - _S_ _, δ_ ) _−_ _ψ_ [ˆ] ( _δ_ ) _}/σ_ ˆ( _δ_ )
� �


over draws of the multipliers ( _ξ_ 1 _, ..., ξ_ _n_ ) (conditional on the sample data **Z** 1 _, ...,_ **Z** _n_ ), which
are iid random variables with mean zero and unit variance that are independent of the
sample. Typically one uses either Gaussian or Rademacher multipliers (i.e., P( _ξ_ = 1) =
P( _ξ_ = _−_ 1) = 0 _._ 5); we use Rademacher multipliers because they gave better performance
in simulations. The next theorem states that this approximation works under the same
assumptions from Theorem 3.


**Theorem 4.** _Let_ ˆ _c_ _α_ _denote the_ 1 _−_ _α quantile (conditional on the data) of the supremum_
_of the multiplier bootstrap process, i.e.,_



_√n_ P _n_
�����



�



_ϕ_ ( **Z** ; ˆ _**η**_ _-S_ _, δ_ ) _−_ _ψ_ [ˆ] ( _δ_ ) ˆ

ˆ _≥_ _c_ _α_ **Z** 1 _, ...,_ **Z** _n_
_σ_ ( _δ_ ) ���

� �������



�



P



sup
_δ∈D_

�



_ξ_



= _α_



_where_ ( _ξ_ 1 _, ..., ξ_ _n_ ) _are iid Rademacher random variables independent of the sample. Then,_
_under the same conditions from Theorem 3,_


ˆ
P _ψ_ ( _δ_ ) _−_ _[c]_ [ˆ] _[α]_ _[σ]_ [ˆ][(] _[δ]_ [)] _≤_ _ψ_ ( _δ_ ) _≤_ _ψ_ [ˆ] ( _δ_ ) + _[c]_ [ˆ] _[α]_ _[σ]_ [ˆ][(] _[δ]_ [)] _, for all δ ∈D_ = 1 _−_ _α_ + _o_ (1) _._
� ~~_√n_~~ ~~_√n_~~ �


14


The proof of Theorem 4 is given in Section 8.5 of the Appendix, and follows by linking the multiplier bootstrap process to the same Gaussian process G to which the scaled
estimator _ψ_ [ˆ] ( _δ_ ) converges. As mentioned above, the multiplier bootstrap only requires
simulating the multipliers _ξ_ and not re-estimating the nuisance functions, so it is straightforward and fast to implement. We include an implementation in the R function given in
Section 8.6 of the Appendix, as well as in the `npcausal` R package available at GitHub
(github.com/ehkennedy/npcausal).

Given the above uniform confidence band, we can test the null hypothesis of no incremental intervention effect


_H_ 0 : _ψ_ ( _δ_ ) = E( _Y_ ) for all _δ ∈D,_


by simply checking whether a (1 _−_ _α_ ) band contains a straight line over _D_ . In other words
we can compute a p-value as


_p_ ˆ = sup _α_ : inf _{ψ_ [ˆ] ( _δ_ ) _−_ _c_ ˆ _α_ _σ_ ˆ( _δ_ ) _/_ _[√]_ _n}_ _._
� _δ∈D_ _[{]_ [ ˆ] _[ψ]_ [(] _[δ]_ [) + ˆ] _[c]_ _[α]_ _[σ]_ [ˆ][(] _[δ]_ [)] _[/][√][n][} ≥]_ [sup] _δ∈D_ �


Note that the condition in the above set corresponds to failing to reject _H_ 0 at level _α_, since
there is space for a straight line between the smallest upper confidence limit and largest
lower confidence limit. We will necessarily fail to reject at level _α_ = 0 since this amounts
to an infinitely wide confidence band, and the p-value is the largest _α_ at which we fail to
reject (i.e., the p-value is small if we reject even for wide bands, and large if we need to
move to narrower bands or never reject).
Interestingly, the hypothesis we test above lies in a middle ground between Fisher’s null
of no individual effect and Neyman’s null of no average effect. _H_ 0 is a granular hypothesis
perhaps closer to Fisher’s null than Neyman’s, but it can still be tested nonparametrically
and in a longitudinal superpopulation framework. This is in contrast to common tests of
Fisher’s null that operate under additive effect hypotheses and are limited to point exposures (Rosenbaum 2002). Thus tests of the null _H_ 0 can be useful in general settings,
independent of any interest in pursuing incremental intervention effects or avoiding positivity assumptions.

### **5 Illustrations**

#### **5.1 Simulation Study**


Here we explore finite-sample properties via simulation, based on the simulation setup used
by Kang & Schafer (2007). In particular we consider their model


( _X_ 1 _, X_ 2 _, X_ 3 _, X_ 4 ) _∼_ _N_ ( **0** _,_ **I** ) _,_
P( _A_ = 1 _|_ **X** ) = expit( _−X_ 1 + 0 _._ 5 _X_ 2 _−_ 0 _._ 25 _X_ 3 _−_ 0 _._ 1 _X_ 4 )
( _Y |_ **X** _, A_ ) _∼_ _N_ _{µ_ ( **X** _, A_ ) _,_ 1 _}_


where the regression function is given by _µ_ ( **x** _, a_ ) = 200 + _a{_ 10 + 13 _._ 7(2 _x_ 1 + _x_ 2 + _x_ 3 + _x_ 4 ) _}_ .
This simulation setup is known to yield variable propensity scores that can degrade the
performance of weighting-based estimators.


15


We considered three estimators in our simulation: a plug-in estimator given by



_,_
�



ˆ
_ψ_ _pi_ ( _δ_ ) = P _n_



_δπ_ ˆ( **X** ) _µ_ ˆ( **X** _,_ 1) + _{_ 1 _−_ _π_ ˆ( **X** ) _}µ_ ˆ( **X** _,_ 0)
� _δπ_ ˆ( **X** ) + 1 _−_ _π_ ˆ( **X** )



along with the inverse-probability-weighted (IPW) estimator and proposed efficient estimator described in Sections 4.1–4.2. We further considered four versions of each these

estimators, depending on how the nuisance functions were estimated: correct parametric
models, misspecified parametric models based on transformed covariates **X** _[∗]_ (using the
same covariate transformations as Kang & Schafer (2007)), and nonparametric estimation
(using original or transformed covariates). For nonparametric estimation we used the crossvalidation-based Super Learner ensemble (van der Laan _et al._ 2007) to combine generalized
additive models, multivariate adaptive regression splines, support vector machines, and
random forests, along with parametric models (with and without interactions, and with
terms selected stepwise via AIC). Regardless of estimator (plug-in, IPW, or proposed), for
nonparametric nuisance estimation we used sample splitting as described in Section 4.2
with _K_ = 2 splits.
Estimator performance was assessed via integrated bias and root-mean-squared error



_J_
�

_j_ =1



_I_



1

_J_
�



1 _/_ 2
ˆ 2
_ψ_ _j_ ( _δ_ _i_ ) _−_ _ψ_ ( _δ_ _i_ )
� �
�



_I_
�


_i_ =1



�
bias = [1]

_I_



_I_
�


_i_ =1



��� _J_ 1



_J_
�



ˆ � _√_ ~~_n_~~

� _ψ_ _j_ ( _δ_ _i_ ) _−_ _ψ_ ( _δ_ _i_ )��� _,_ RMSE = _I_

_j_ =1



across _J_ = 500 simulations and _I_ = 100 values of _δ_ equally spaced (on the log scale)
between exp( _−_ 2 _._ 3) _≈_ 0 _._ 1 and exp(2 _._ 3) _≈_ 10. Results are given in Figure 2.
In each setting, the proposed estimator performed as well or better than the plug-in and
IPW versions. When the nuisance functions were estimated with correct parametric models,
all methods gave small bias and RMSE, with the plug-in and proposed estimators slightly
outperforming the IPW estimator in terms of RMSE. Under parametric misspecification,
bias and RMSE were amplified for all estimators and the plug-in fared worst. A more
interesting (but expected) story appeared with nonparametric nuisance estimation. There,
the plug-in and IPW estimators show large bias and RMSE, since they are not expected to
converge at _[√]_ ~~_n_~~ rates; in contrast, the proposed efficient estimator essentially matches its
behavior when constructed based on correct parametric models (with only a slight loss in
RMSE). This is indicative of the fact that the proposed estimator only requires _n_ [1] _[/]_ [4] rates
on nuisance estimation to achieve full efficiency and in general has second-order bias. This
behavior appears to hold in our simulations even for nonparametric estimation using **X** _[∗]_,
i.e., when the true model is not used directly.
We also assessed the uniform coverage of our proposed multiplier bootstrap confidence
bands (as usual, we say a band covers if it contains the true curve entirely for all _δ ∈D_ ).
Results are given in Table 1. As expected, coverage is very poor when nuisance functions
are estimated with misspecified parametric models. Coverage was near the nominal level
(95%) in large samples as long as nuisance functions were estimated with correct parametric
models or nonparametrically using the non-transformed covariates **X** (coverage was slightly
diminished for nonparametric nuisance estimation based on the misspecified **X** _[∗]_ ).


16


**n=500**

|Col1|Col2|Col3|Col4|n=|
|---|---|---|---|---|
||Proposed<br>Plug−in<br>IPW|Proposed<br>Plug−in<br>IPW|||
||||||
||||||
||||||
||||||
||||||
||||||
||||||



Cor P Mis P Cor NP Mis NP

|Col1|Proposed<br>Plug−in<br>IPW|
|---|---|
|||
|||
|||
|||



Cor P Mis P Cor NP Mis NP



**n=1000**

|Col1|Proposed<br>Plug−in<br>IPW|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
||||||
||||||
||||||
||||||
||||||
||||||



Cor P Mis P Cor NP Mis NP

|Col1|Proposed<br>Plug−in<br>IPW|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
||||||
||||||
||||||



Cor P Mis P Cor NP Mis NP



**n=5000**

|Col1|Proposed<br>Plug−in<br>IPW|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
||||||
||||||
||||||
||||||
||||||
||||||



Cor P Mis P Cor NP Mis NP

|Col1|Proposed<br>Plug−in<br>IPW|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
||||||
||||||
||||||
||||||
||||||



Cor P Mis P Cor NP Mis NP



**Figure 2:** Integrated bias and root-mean-squared-error (RMSE) across 500 simulations.
(IPW = inverse-probability-weighted; P/NP = parametric/nonparametric nuisance estimation based on covariates **X** (Cor) or transformed version **X** _[∗]_ (Mis).)


**Table 1:** Coverage of proposed uniform 95% confidence band across 500 simulations.


Sample Coverage (%) for setting:
size _n_ Cor P Mis P Cor NP Mis NP

500 92.4 77.0 93.0 88.0

1000 95.2 67.6 95.6 92.4

5000 94.8 12.4 94.2 89.4

#### **5.2 Application**


Here we illustrate the use of incremental intervention effects with a reanalysis of the National Longitudinal Survey of Youth 1997 data used by Huebner (2005), Bacak & Kennedy
(2015), and others to study the effects of incarceration on marriage. Incarceration is a
colossal industry in the United States, with over 2.3 million people currently confined in a
correctional facility and at least twice that number held on probation or parole (Wagner &
Rabuy 2016). There is a large literature on unintended effects of this mass incarceration,
with numerous studies pointing to negative impacts on various aspects of employment,
health, social ties, psychology, and more (Clear 2009; Pattillo _et al._ 2004). Effects of incar

17


ceration on marriage are important since marriage is expected to yield, for example, better
family and social support, better outcomes for children, and less recidivism, among other
benefits (Clear 2009; Huebner 2005). Bacak & Kennedy (2015) were the first to study this
question while specifically accounting for time-varying confounders, such as employment
and earnings, and we refer there for more motivation and background.
The National Longitudinal Survey of Youth 1997 data consists of yearly measures across
14 timepoints, from 1997 to 2010, for participants who were 12–16 years old at the initial
survey. The data include demographic information (e.g., age, race, gender, parent’s education), various delinquency indicators (e.g., age at first sex, measures of drug use and
gang membership, delinquency scores), as well as numerous time-varying measures (e.g.,
employment, earnings, marriage and incarceration history). Following Bacak & Kennedy
(2015), we use the final 10 timepoints from 2001–2010, restrict the analysis to the 4781 individuals with a non-zero delinquency score at baseline, and use as outcome _Y_ the indicator
of marriage at the end of the study (i.e., in 2010).
Bacak & Kennedy (2015) used a standard marginal structural model approach to study
effects of static incarceration trajectories, which has some limitations. First, it requires
a parametric model to describe how incarceration trajectories affect marriage rates. In
particular Bacak & Kennedy (2015) used E( _Y_ ~~_[a]_~~ _[T]_ ) = expit( _β_ 0 + _β_ 1 � _t_ _[a]_ _[t]_ [), which only allows]



particular Bacak & Kennedy (2015) used E( _Y_ ~~_[a]_~~ _[T]_ ) = expit( _β_ 0 + _β_ 1 � _t_ _[a]_ _[t]_ [), which only allows]

marriage prevalence to depend on total time spent incarcerated. This kind of assumption
is very common in practice but is quite restrictive, especially since a saturated structural
model in this case would have 2 [10] = 1024 parameters, instead of only two. Hence the
data only inform 2 _/_ 1024 = 0 _._ 2% of the possible parameter values. In fact if the model
is slightly elaborated, e.g., to E( _Y_ ~~_[a]_~~ _[T]_ ) = expit( _β_ 0 + [�] _t_ _[β]_ [1] _[t]_ _[a]_ _[t]_ [) so that] _[ β]_ [1] [ can vary with]



is slightly elaborated, e.g., to E( _Y_ ~~_[a]_~~ _[T]_ ) = expit( _β_ 0 + [�] _t_ _[β]_ [1] _[t]_ _[a]_ _[t]_ [) so that] _[ β]_ [1] [ can vary with]

time, then a standard weighting estimator fails and no coefficient estimates can be found.
Another limitation is that Bacak & Kennedy (2015) used parametric inverse probability
weighting to estimate ( _β_ 0 _, β_ 1 ) (partly for pedagogic purposes), but this is both inefficient
and likely biased due to propensity score model misspecification. Perhaps most importantly,
a standard marginal structural model setup requires imagining sending _all_ or _none_ of the
study participants to prison at each time. However, positivity is likely violated here since
some individuals may be necessarily incarcerated at some times (e.g., due to multiple-year
sentences) or have essentially zero chance of incarceration (based on demographic or other
characteristics). These limitations are not at all unique to the analysis of Bacak & Kennedy
(2015), but instead are common to many observational marginal structural model analyses;
we build on their analysis by instead estimating incremental incarceration effects, which
require neither any parametric models nor any positivity assumptions.
Specifically we estimated the incremental effect curve _ψ_ ( _δ_ ), which in this setting represents the marriage prevalence at the end of the study if the odds of incarceration were
multiplied by factor _δ_ . We used Random Forests (via the `ranger` package in R) to estimate
all nuisance functions _π_ _t_ and _m_ _t_ as described in Algorithm 1 (with _K_ = 10-fold sample
splitting), and computed pointwise and uniform confidence bands as in Sections 4.3 and
4.4 (with 10,000 bootstrap replications). Results are shown in Figure 3.
We find strong evidence (assuming no unmeasured confounding and consistency) that
incarceration negatively impacts marriage rates. First, we reject the null hypothesis of
no incremental effect of incarceration on marriage ( _p_ = 0 _._ 049) over the range _δ ∈_ [0 _._ 2 _,_ 5].



18


|Col1|Estimate<br>Pointwise 95% CI<br>Uniform 95% CI|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
||||||||
||||||||
||||||||
||Estimate<br>Pointwise 95% CI<br>Uniform 95% CI|Estimate<br>Pointwise 95% CI<br>Uniform 95% CI|Estimate<br>Pointwise 95% CI<br>Uniform 95% CI|Estimate<br>Pointwise 95% CI<br>Uniform 95% CI|Estimate<br>Pointwise 95% CI<br>Uniform 95% CI|Estimate<br>Pointwise 95% CI<br>Uniform 95% CI|
||Estimate<br>Pointwise 95% CI<br>Uniform 95% CI||||||


0.2 0.5 1 2 5


Incarceration odds ratio δ


**Figure 3:** Estimated marriage prevalence 10 years post-baseline, if the incarceration odds
were multiplied by factor _δ_, with pointwise and uniform 95% confidence bands.


More specifically we estimate that, if incarceration odds were increased proportionally for
all individuals, marriage prevalence would drop from P _n_ ( _Y_ ) = 29 _._ 4% observationally to
28.1% if the odds doubled (OR=0.94, 95% CI: 0.87–1.00), and to 23.6% if the odds were
multiplied four-fold (OR=0.74, 95% CI: 0.59–0.91). Conversely, we estimate that marriage
prevalence would only increase to 29.7% if the odds of incarceration were halved (OR=1.01,
95% CI: 0.95–1.08); the prevalence and odds ratio are the same if the odds were quartered.
These results suggest that marriage rates might be more affected by increased rather than
decreased incarceration (i.e., the curve in Figure 3 is nonlinear, with larger slope for _δ >_ 1).
This analysis provides considerably more nuance than a simple marginal structural model
fit, and requires none of the parametric and positivity assumptions.

### **6 Discussion**


In this paper we have proposed incremental intervention effects, which are based on shifting
propensity scores rather than setting treatment values. We showed that these effects can
be identified and estimated without any positivity or parametric assumptions, established
general efficiency theory, and constructed influence-function-based estimators that yield
fast rates of convergence even when based on flexible nonparametric regression tools. We


19


also developed an approach for uniform inference and a new test of no treatment effect,
and applied the methods in a longitudinal study of incarceration effects on marriage.
There are a few caveats to our developments that are worth mentioning. First, we expect
incremental intervention effects to play a more descriptive than prescriptive role compared
to other approaches. Specifically, they give an interpretable picture of what would happen
if exposure were increased or decreased in a natural way, but will likely be less useful for
informing specific treatment decisions. For example in our analysis from Section 5.2 the
goal was to better understand the overall societal effects of mass incarceration; in cases
where the goal is to learn how to best assign treatment, methods for optimal treatment
regime estimation will likely be more relevant. However, note that it is certainly possible
to estimate the optimal incremental regime _q_ _t_ ( **h** _t_ ; _δ_ _[∗]_ _, π_ _t_ ) for _δ_ _[∗]_ = arg max _δ_ E( _Y_ **[Q]** [(] _[δ]_ [)] ); so in
theory incremental effects could be used to construct specific treatment decision rules.
Another caveat is that, in favor of computational efficiency, we have bypassed concerns
about model compatibility when estimating the pseudo-regression functions _m_ _t_ . It can be
difficult to formulate models for all the _m_ _t_ functions that are compatible with each other,
since _m_ _t_ has a complicated dependence on _m_ _t_ +1 (as well as the propensity scores _π_ _t_ +1 and
covariate densities). To make these estimators fully compatible, we would need to model
the conditional densities of the (high-dimensional) covariates and construct ˆ _m_ _t_ based on the
non-recursive expression in Theorem 2. However, we feel that if flexible enough estimators
for _m_ _t_ are used, then model incompatibility will likely not be a major concern in practice,
particularly relative to the computational benefits. This issue also arises in estimating
standard longitudinal causal effects (Murphy _et al._ 2001; Scharfstein _et al._ 1999).
In future work we plan to pursue various extensions of incremental intervention effects.
For example, it will be important to consider (i) interventions with increment parameters _δ_ = _δ_ ( **h** _t_ ) that depend on time and past covariate history, (ii) estimation of how
mean outcomes under different interventions vary with covariates (effect modification),
(iii) extensions for settings with multivalued treatments and/or censored outcomes, and
(iv) increment parameters based on risk ratios or other shifts, rather than odds ratios.


20


### **7 References**

1. Bacak, V. & Kennedy, E. H. Marginal structural models: An application to incarceration and marriage during young adulthood. _Journal of Marriage and Family_ **77,**
112–125 (2015).


2. Belloni, A., Chernozhukov, V., Chetverikov, D. & Wei, Y. Uniformly valid postregularization confidence regions for many functional parameters in Z-estimation framework. _arXiv preprint arXiv:1512.07619_ (2015).


3. Bickel, P. J., Klaassen, C. A., Ritov, Y. & Wellner, J. A. _Efficient and Adaptive_
_Estimation for Semiparametric Models_ (Johns Hopkins University Press, 1993).


4. Cain, L. E. _et al._ When to start treatment? A systematic approach to the comparison
of dynamic regimes using observational data. _The International Journal of Biostatis-_
_tics_ **6,** 1–24 (2010).


5. Chernozhukov, V. _et al._ Double machine learning for treatment and causal parameters.
_arXiv preprint arXiv:1608.00060,_ 1–37 (2016).


6. Chernozhukov, V., Chetverikov, D. & Kato, K. Gaussian approximation of suprema
of empirical processes. _The Annals of Statistics_ **42,** 1564–1597 (2014).


7. Clear, T. R. _Imprisoning communities: How mass incarceration makes disadvantaged_
_neighborhoods worse_ (Oxford University Press, 2009).


8. D´ıaz, I. & van der Laan, M. J. Assessing the causal effect of policies: an example
using stochastic interventions. _The International Journal of Biostatistics_ **9,** 161–174
(2013).


9. D´ıaz, I. & van der Laan, M. J. Population intervention causal effects based on stochastic interventions. _Biometrics_ **68,** 541–549 (2012).


10. Dud´ık, M., Erhan, D., Langford, J. & Li, L. Doubly robust policy evaluation and
optimization. _Statistical Science_ **29,** 485–511 (2014).


11. Gin´e, E. & Zinn, J. Some limit theorems for empirical processes. _The Annals of Prob-_
_ability_ **12,** 929–989 (1984).


12. Haneuse, S & Rotnitzky, A. Estimation of the effect of interventions that modify the
received treatment. _Statistics in Medicine_ **32,** 5260–5277 (2013).


13. Huebner, B. M. The effect of incarceration on marriage and work over the life course.
_Justice Quarterly_ **22,** 281–303 (2005).


14. Kang, J. D. & Schafer, J. L. Demystifying double robustness: A comparison of alternative strategies for estimating a population mean from incomplete data. _Statistical_
_Science_ **22,** 523–539 (2007).


15. Kennedy, E. H. _Semiparametric theory and empirical processes in causal inference_
in _Statistical Causal Inferences and Their Applications in Public Health Research_
(Springer, 2016), 141–167.


21


16. Kennedy, E. H., Ma, Z., McHugh, M. D. & Small, D. S. Nonparametric methods
for doubly robust estimation of continuous treatment effects. _Journal of the Royal_
_Statistical Society: Series B_ **79,** 1229–1245 (2017).


17. Kennedy, E. H., Lorch, S. A. & Small, D. S. Robust causal inference with continuous
instruments using the local instrumental variable curve. _arXiv preprint arXiv:1607.02566_
(2016).


18. Moore, K. L., Neugebauer, R., van der Laan, M. J. & Tager, I. B. Causal inference in
epidemiological studies with strong confounding. _Statistics in Medicine_ **31,** 1380–1404
(2012).


19. Murphy, S. A. Optimal dynamic treatment regimes. _Journal of the Royal Statistical_
_Society: Series B_ **65,** 331–355 (2003).


20. Murphy, S. A., van der Laan, M. J. & Robins, J. M. Marginal mean models for dynamic
regimes. _Journal of the American Statistical Association_ **96,** 1410–1423 (2001).


21. Pattillo, M., Western, B. & Weiman, D. _Imprisoning America: The social effects of_
_mass incarceration_ (Russell Sage Foundation, 2004).


22. Pearl, J. _Causality: Models, Reasoning, & Inference_ (Cambridge Univ. Press, 2009).


23. Robins, J. M. A new approach to causal inference in mortality studies with a sustained exposure period: application to control of the healthy worker survivor effect.
_Mathematical Modelling_ **7,** 1393–1512 (1986).


24. Robins, J. M. _Marginal structural models versus structural nested models as tools for_
_causal inference_ in _Statistical Models in Epidemiology, the Environment, and Clinical_
_Trials_ (Springer, 2000), 95–133.


25. Robins, J. M. _Optimal structural nested models for optimal sequential decisions_ in
_Proceedings of the Second Seattle Symposium in Biostatistics_ (2004), 189–326.


26. Robins, J. M., Hern´an, M. A. & Siebert, U. _Effects of multiple interventions_ in _Com-_
_parative Quantification of Health Risks_ (Citeseer, 2004), 2191–2230.


27. Robins, J. M., Orellana, L. & Rotnitzky, A. Estimation and extrapolation of optimal
treatment and testing strategies. _Statistics in Medicine_ **27,** 4678–4721 (2008).


28. Robins, J. M., Hern´an, M. A. & Brumback, B. Marginal structural models and causal
inference in epidemiology. _Epidemiology_ **11,** 550–560 (2000).


29. Rosenbaum, P. R. Covariance adjustment in randomized experiments and observational studies (with discussion). _Statistical Science_ **17,** 286–327 (2002).


30. Rosenbaum, P. R. & Rubin, D. B. The central role of the propensity score in observational studies for causal effects. _Biometrika_ **70,** 41–55 (1983).


31. Rubin, D. B. Estimating causal effects of treatments in randomized and nonrandomized studies. _Journal of Educational Psychology_ **66,** 688–701 (1974).


32. Scharfstein, D. O., Rotnitzky, A. & Robins, J. M. Adjusting for nonignorable dropout using semiparametric nonresponse models. _Journal of the American Statistical_
_Association_ **94,** 1096–1120 (1999).


22


33. Taubman, S. L., Robins, J. M., Mittleman, M. A. & Hern´an, M. A. Intervening on
risk factors for coronary heart disease: an application of the parametric g-formula.
_International Journal of Epidemiology_ **38,** 1599–1611 (2009).


34. Tian, J. Identifying dynamic sequential plans. _Proceedings of the Twenty-Fourth Con-_
_ference on Uncertainty in Artificial Intelligence_ (2008).


35. Tsiatis, A. A. _Semiparametric Theory and Missing Data_ (Springer, 2006).


36. van der Laan, M. J. & Petersen, M. L. Causal effect models for realistic individualized
treatment and intention to treat rules. _The International Journal of Biostatistics_ **3,**
1–52 (2007).


37. van der Laan, M. J. & Robins, J. M. _Unified Methods for Censored Longitudinal Data_
_and Causality_ (Springer, 2003).


38. van der Laan, M. J., Polley, E. C. & Hubbard, A. E. Super learner. _Statistical Appli-_
_cations in Genetics and Molecular Biology_ **6,** 1–21 (2007).


39. van der Vaart, A. W. _Asymptotic Statistics_ (Cambridge University Press, 2000).


40. van der Vaart, A. W. Semiparametric statistics. _In: Lectures on Probability Theory_
_and Statistics,_ 331–457 (2002).


41. van der Vaart, A. W. & Wellner, J. A. _Weak Convergence and Empirical Processes_
(Springer, 1996).


42. Wagner, P. & Rabuy, B. Mass incarceration: The whole pie 2016. _Retrieved from the_
_Prison Policy Initiative Website: www.prisonpolicy.org/reports/pie2016.html_ (2016).


43. Young, J. G., Cain, L. E., Robins, J. M., O’Reilly, E. J. & Hern´an, M. A. Comparative effectiveness of dynamic treatment regimes: an application of the parametric
g-formula. _Statistics in Biosciences_ **3,** 119–143 (2011).


44. Young, J. G., Hern´an, M. A. & Robins, J. M. Identification, estimation and approximation of risk under interventions that depend on the natural value of treatment
using observational data. _Epidemiologic Methods_ **3,** 1–19 (2014).


45. Zheng, W. & van der Laan, M. J. Asymptotic theory for cross-validated targeted
maximum likelihood estimation. _UC Berkeley Division of Biostatistics Working Paper_
_Series_ **Paper 273,** 1–58 (2010).


23


### **8 Appendix**

#### **8.1 Proof of Theorem 1**

First we give a useful identification result for general stochastic intervention effects.


**Lemma 1.** _Let_ **Q** = ( _Q_ 1 _, ..., Q_ _T_ ) _denote a general stochastic intervention in which treat-_
_ment at time t is randomly assigned according to distribution function Q_ _t_ ( _a_ _t_ _|_ **h** _t_ ) _. Under_
_Assumptions 1–2, and if (weak) positivity holds in the sense that_


_d_ P( _a_ _t_ _|_ **h** _t_ ) = 0 = _⇒_ _dQ_ _t_ ( _a_ _t_ _|_ **h** _t_ ) = 0


_then the mean outcome_ E( _Y_ **[Q]** ) _under the intervention is identified by_



_ψ_ _[∗]_ ( **Q** ) =
� _A_



E( _Y |_ **X** _T_ = ~~**x**~~ _T_ _, A_ _T_ = ~~_a_~~ _T_ )

� _X_



_T_
� _dQ_ _t_ ( _a_ _t_ _|_ **h** _t_ ) _d_ P( **x** _t_ _|_ **h** _t−_ 1 _, a_ _t−_ 1 ) _,_


_t_ =1



_where A_ = _A_ 1 _× . . . A_ _T_ _and X_ = _X_ 1 _. . . X_ _T_ _._


_Proof._ This essentially follows by the g-formula of Robins (1986). Let underbars denote
the future of a sequence so that for example **Q** ~~_t_~~ = ( _Q_ _t_ _, ..., Q_ _T_ ). Then we have the recursion


E( _Y_ [(] ~~_[a]_~~ _[t][−]_ [1] _[,]_ **[Q]** ~~_[t]_~~ [)] _|_ **H** _t−_ 1 _, A_ _t−_ 1 ) = E _{_ E( _Y_ [(] ~~_[a]_~~ _[t][−]_ [1] _[,]_ **[Q]** ~~_[t]_~~ [)] _|_ **H** _t_ _, Q_ _t_ ) _|_ **H** _t−_ 1 _, A_ _t−_ 1 _}_


= E( _Y_ [(] ~~_[a]_~~ _[t][−]_ [1] _[,q]_ _[t]_ _[,]_ **[Q]** _[t]_ [+1] [)] _|_ **H** _t_ = **h** _t_ _, Q_ _t_ = _q_ _t_ ) _dQ_ _t_ ( _q_ _t_ _|_ **h** _t_ ) _d_ P( **x** _t_ _|_ **h** _t−_ 1 _, a_ _t−_ 1 )
�


= E( _Y_ [(] ~~_[a]_~~ _[t][−]_ [1] _[,q]_ _[t]_ _[,]_ **[Q]** _[t]_ [+1] [)] _|_ **H** _t_ = **h** _t_ _, A_ _t_ = _q_ _t_ ) _dQ_ _t_ ( _q_ _t_ _|_ **h** _t_ ) _d_ P( **x** _t_ _|_ **h** _t−_ 1 _, a_ _t−_ 1 )
�


= E( _Y_ [(] ~~_[a]_~~ _[t]_ _[,]_ **[Q]** ~~_[t]_~~ [+1] [)] _|_ **H** _t_ = **h** _t_ _, A_ _t_ = _a_ _t_ ) _dQ_ _t_ ( _a_ _t_ _|_ **h** _t_ ) _d_ P( **x** _t_ _|_ **h** _t−_ 1 _, a_ _t−_ 1 )
�


for _t_ = 1 _, ..., T_, where the first equality follows by iterated expectation, the second by definition, the third since _Q_ _t_ _⊥⊥_ _Y_ **[Q]** _|_ **H** _t_ (by definition) along with exchangeability (Assumption
2), and the fourth by simply rewriting the index _q_ _t_ as _a_ _t_ . The weak positivity condition
is required so that the above outer expectation is well-defined (that the inner expectation
may not be is fine since, by positivity, in such cases the multiplier _dQ_ _t_ will be zero).
Therefore applying the above _T_ times yields



E( _Y_ **[Q]** ) =
� _A_ 1


=
�



E( _Y_ [(] _[a]_ [1] _[,]_ **[Q]** ~~[2]~~ [)] _|_ **X** 1 = **x** 1 _, A_ 1 = _a_ 1 ) _dQ_ 1 ( _a_ 1 _|_ **x** 1 ) _d_ P( **x** 1 )

� _X_ 1



E( _Y_ [(] ~~_[a]_~~ [2] _[,]_ **[Q]** ~~[3]~~ [)] _|_ **H** 2 = **h** 2 _, A_ 2 = _a_ 2 )



2
� _dQ_ _t_ ( _a_ _t_ _|_ **h** _t_ ) _d_ P( **x** _t_ _|_ **h** _t−_ 1 _, a_ _t−_ 1 )


_t_ =1



_A_ 1 _×A_ 2



�

_X_ 1 _×X_ 2



=
� _A_


=
� _A_



E( _Y_ ~~_[a]_~~ _[T]_ _|_ **X** _T_ = ~~**x**~~ _T_ _, A_ _T_ = ~~_a_~~ _T_ )

� _X_



_T_
� _dQ_ _t_ ( _a_ _t_ _|_ **h** _t_ ) _d_ P( **x** _t_ _|_ **h** _t−_ 1 _, a_ _t−_ 1 )


_t_ =1



E( _Y |_ **X** _T_ = ~~**x**~~ _T_ _, A_ _T_ = ~~_a_~~ _T_ )

� _X_



_T_
� _dQ_ _t_ ( _a_ _t_ _|_ **h** _t_ ) _d_ P( **x** _t_ _|_ **h** _t−_ 1 _, a_ _t−_ 1 )


_t_ =1



where the last equality follows by consistency (Assumption 1).


1


Now Theorem 1 follows from Lemma 1, letting



and noting that



_dQ_ _t_ ( _a_ _t_ _|_ **h** _t_ ) = _[a]_ _[t]_ _[δπ]_ _[t]_ [(] **[h]** _[t]_ [)][ + ][(][1] _[ −]_ _[a]_ _[t]_ [)] _[{]_ [1] _[ −]_ _[π]_ _[t]_ [(] **[h]** _[t]_ [)] _[}]_

_δπ_ _t_ ( **h** _t_ ) + 1 _−_ _π_ _t_ ( **h** _t_ )


_π_ _t_ ( **h** _t_ ) = 0 = _⇒_ _dQ_ _t_ (1 _|_ **h** _t_ ) = 0
_π_ _t_ ( **h** _t_ ) = 1 = _⇒_ _dQ_ _t_ (0 _|_ **h** _t_ ) = 0



so that the weak positivity condition is automatically satisfied by our choice of _dQ_ _t_ .

#### **8.2 Proof of Theorem 2**


First we derive the efficient influence function for a general stochastic intervention effect
when the intervention distribution _Q_ does not depend on the observed data distribution P.


**Lemma 2.** _Suppose_ **Q** _is a known stochastic intervention not depending on_ P _. Define_



_m_ _t_ ( **h** _t_ _, a_ _t_ ) = _µ_ ( **h** _T_ _, a_ _T_ )
� _R_ _t_



_T_
� _dQ_ _s_ ( _a_ _s_ _|_ **h** _s_ ) _d_ P( **x** _s_ _|_ **h** _s−_ 1 _, a_ _s−_ 1 )


_s_ = _t_ +1



_for t_ = 0 _, ..., T −_ 1 _and R_ _t_ = ( _H_ _T_ _× A_ _T_ ) _\ H_ _t_ _, and let m_ _T_ ( **h** _T_ _, a_ _T_ ) = _µ_ ( **h** _T_ _, a_ _T_ ) _and_
_m_ _T_ +1 ( **h** _T_ +1 _, a_ _T_ +1 ) = _Y . Then the efficient influence function for ψ_ _[∗]_ ( **Q** ) = _m_ 0 _is_



_T_
�


_t_ =0



_t_
_m_ _t_ +1 ( **H** _t_ +1 _, a_ _t_ +1 ) _dQ_ _t_ +1 ( _a_ _t_ +1 _|_ **H** _t_ +1 ) _−_ _m_ _t_ ( **H** _t_ _, A_ _t_ ) �
_A_ _t_ +1 �



��



_s_ =0



_dQ_ _s_ ( _A_ _s_ _|_ **H** _s_ )

_d_ P( _A_ _s_ _|_ **H** _s_ )



=



_T_
�


_t_ =1



��




_[|]_ **[ H]** _[t]_ [)]
_m_ _t_ ( **H** _t_ _, a_ _t_ ) _dQ_ _t_ ( _a_ _t_ _|_ **H** _t_ ) _−_ _m_ _t_ ( **H** _t_ _, A_ _t_ ) _[d][Q]_ _[t]_ [(] _[A]_ _[t]_
_A_ _t_ _d_ P( _A_ _t_ _|_ **H** _t_ )



_d_ P( _A_ _t_ _|_ **H** _t_ )



_t−_ 1
�
� _s_ =0



_dQ_ _s_ ( _A_ _s_ _|_ **H** _s_ )

_d_ P( _A_ _s_ _|_ **H** _s_ )



_T_


+

�


_s_ =1



_dQ_ _s_ ( _A_ _s_ _|_ **H** _s_ )

_d_ P( _A_ _s_ _|_ **H** _s_ ) _[Y][ −]_ _[ψ]_ _[∗]_ [(] **[Q]** [)]



_where we define dQ_ _T_ +1 = 1 _and dQ_ 0 ( _a_ 0 _|_ **h** 0 ) _/d_ P( _a_ 0 _|_ **h** 0 ) = 1 _._


**Lemma 3.** _Suppose_ **Q** _depends on_ P _, and let {_ 1( **H** _t_ = **h** _t_ ) _/d_ P( **h** _t_ ) _}φ_ _t_ ( **H** _t_ _, A_ _t_ ; _a_ _t_ ) _denote_
_the efficient influence function for dQ_ _t_ ( _a_ _t_ _|_ **h** _t_ ) _. Then the efficient influence for_ E( _Y_ **[Q]** )
_allowing_ **Q** _to depend on_ P _is given by_



_t−_ 1
�
� _s_ =0



_φ_ _t_ ( **H** _t_ _, A_ _t_ ; _a_ _t_ ) _m_ _t_ ( **H** _t_ _, a_ _t_ ) _dν_ ( _a_ _t_ )
_A_ _t_

��



_ϕ_ _[∗]_ ( **Q** ) +



_T_
�


_t_ =1



_dQ_ _s_ ( _A_ _s_ _|_ **H** _s_ )

_d_ P( _A_ _s_ _|_ **H** _s_ )



_where ϕ_ _[∗]_ ( **Q** ) _denotes the efficient influence function from Lemma 2 under an intervention_
**Q** _not depending on_ P _, and ν is a dominating measure for the distribution of A_ _t_ _._


2


The proofs of Lemmas 2 and 3 are based on chain rule arguments stemming from
the fact that the efficient influence function is a pathwise derivative. In particular (in
a nonparametric model) the efficient influence function for parameter _ψ_ = _ψ_ (P) is the
function _ϕ_ (P) satisfying



_∂_ _∂_
_ϕ_ (P)
_∂ϵ_ _[ψ]_ [(][P] _[ϵ]_ [)] ��� _ϵ_ =0 [=] � � _∂ϵ_ [log] _[ d]_ [P] _[ϵ]_



���� _ϵ_ =0 _[d]_ [P]



where _{_ P _ϵ_ : _ϵ ∈_ R _}_ is a smooth parametric submodel with P _ϵ_ =0 = P. We omit the proofs
since they are lengthy and not particularly illuminating; however we plan to include them
in a forthcoming paper on general stochastic interventions.


**Lemma 4.** _The efficient influence function for_


_dQ_ _t_ ( _a_ _t_ _|_ **h** _t_ ) = _[a]_ _[t]_ _[δπ]_ _[t]_ [(] **[h]** _[t]_ [)][ + ][(][1] _[ −]_ _[a]_ _[t]_ [)] _[{]_ [1] _[ −]_ _[π]_ [(] **[h]** _[t]_ [)] _[}]_

_δπ_ _t_ ( **h** _t_ ) + 1 _−_ _π_ _t_ ( **h** _t_ )


_is given by {_ 1( **H** _t_ = **h** _t_ ) _/d_ P( **h** _t_ ) _}φ_ _t_ ( **H** _t_ _, A_ _t_ ; _a_ _t_ ) _where φ_ _t_ ( **H** _t_ _, A_ _t_ ; _a_ _t_ ) _equals_


(2 _a_ _t_ _−_ 1) _δ{A_ _t_ _−_ _π_ _t_ ( **H** _t_ ) _}_
_{δπ_ _t_ ( **H** _t_ ) + 1 _−_ _π_ _t_ ( **H** _t_ ) _}_ [2] _[.]_


_Proof._ This result also follows from the chain rule, together with the fact that the efficient
influence function for _π_ _t_ is given by


1( **H** _t_ = **h** _t_ ) _{A_ _t_ _−_ _π_ _t_ ( **h** _t_ ) _}/d_ P( **h** _t_ ) _._

#### **8.3 Z-Estimator Algorithm**


**Algorithm 2** (Z-estimator algorithm) **.** _For each δ:_


_1. Regress A_ _t_ _on_ **H** _t_ _, obtain predicted values_ ˆ _π_ _t_ ( **H** _t_ ) _for each subject/time._


_δA_ _t_ +1 _−A_ _t_
_2. Construct time-dependent weights W_ _t_ = _δπ_ ˆ _t_ ( **H** _t_ )+1 _−π_ ˆ _t_ ( **H** _t_ ) _[for each subject/time.]_


_3. Calculate cumulative product weight_ _W_ [�] _t_ = [�] _[t]_ _s_ =1 _[W]_ _[s]_ _[ for each subject/time.]_


_4. For each time t_ = _T, T −_ 1 _, ...,_ 1 _(starting with R_ _T_ +1 = _Y ):_


_(a) Regress R_ _t_ +1 _on_ ( **H** _t_ _, A_ _t_ ) _, obtain predicted values_ ˆ _m_ _t_ ( **H** _t_ _,_ 1) _and_ ˆ _m_ _t_ ( **H** _t_ _,_ 0) _._

_(b) Construct pseudo-outcome R_ _t_ = _[δ][π]_ [ˆ] _[t]_ [(] **[H]** _[t]_ [)][ ˆ] _[m]_ _[t]_ _δ_ [(] _π_ ˆ **[H]** _t_ ( _[t]_ **H** _[,]_ [1][)+] _t_ )+1 _[{]_ [1] _[−]_ _−_ _[π]_ _π_ [ˆ] ˆ _[t]_ _t_ [(] ( **[H]** **H** _[t]_ _t_ [)] ) _[}]_ [ ˆ] _[m]_ _[t]_ [(] **[H]** _[t]_ _[,]_ [0][)] _._



_5. For each subject compute ϕ_ = _W_ [�] _T_ _Y_ + [�]



_t_ _W_ [�] _t_ _V_ _t_ _R_ _t_ _where V_ _t_ = _[A]_ _[t]_ _[{]_ [1] _[−][π]_ [ˆ] _[t]_ [(] **[H]** _[t]_ [)] _[}][−]_ _−_ [(][1] _δ_ _[−][A]_ _[t]_ [)] _[δ][π]_ [ˆ] _[t]_ [(] **[H]** _[t]_ [)]



_δ/_ (1 _−δ_ ) _._



_6. Set_ _ψ_ [ˆ] _[∗]_ ( _δ_ ) _to be the average of the ϕ values across subjects._


3


#### **8.4 Proof of Theorem 3**

Let _∥f_ _∥_ _D_ = sup _δ∈D_ _|f_ ( _δ_ ) _|_ denote the supremum norm over _D_, and define the processes


�
Ψ _n_ ( _δ_ ) = _[√]_ _n{ψ_ ˆ( _δ_ ) _−_ _ψ_ ( _δ_ ) _}/σ_ ˆ( _δ_ )

�
Ψ _n_ ( _δ_ ) = _[√]_ _n{ψ_ ˆ( _δ_ ) _−_ _ψ_ ( _δ_ ) _}/σ_ ( _δ_ )
Ψ _n_ ( _δ_ ) = G _n_ [ _{ϕ_ ( **Z** ; _**η**_ _, δ_ ) _−_ _ψ_ ( _δ_ ) _}/σ_ ( _δ_ )] = G _n_ _{ϕ_ �( **Z** ; _**η**_ _, δ_ ) _}_


where G _n_ = _[√]_ ~~_n_~~ ~~(~~ P _n_ _−_ P) is the empirical process on the full sample process as usual. Also
let G( _δ_ ) denote the mean-zero Gaussian process with covariance E _{ϕ_ �( **Z** ; _**η**_ _, δ_ 1 ) _ϕ_ �( **Z** ; _**η**_ _, δ_ 2 ) _}_
as in the main text.


In this proof we will show that


Ψ _n_ ( _·_ ) ⇝ G( _·_ ) in _ℓ_ _[∞]_ ( _D_ ) and _∥_ Ψ [�] _n_ _−_ Ψ _n_ _∥_ _D_ = _o_ P (1)


which yields the desired result. The first statement will be true if the influence function � _ϕ_
is a smooth enough function of _δ_, and the second if the nuisance estimators ˆ _**η**_ are consistent
and converging at a sufficiently fast rate.


The first statement follows since the function class ~~_F_~~ ~~_**η**_~~ = _{ϕ_ ( _·_ ; ~~_**η**_~~ ~~_,_~~ _δ_ ) : _δ ∈D}_ is Lipschitz
and thus has a finite bracketing integral for any fixed ~~_**η**_~~ ~~.~~ Recall the _L_ 2 (P) bracketing integral
of class _F_ with envelope _F_ is given by



1
_J_ [ ] ( _F_ ) =
� 0



�



1 + log _N_ [ ] ( _ϵ∥F_ _∥, F, L_ 2 (P)) _dϵ_



where _N_ [ ] ( _ϵ, F, L_ 2 (P)) is the _L_ 2 (P) bracketing number, i.e., the minimum number of _ϵ_
brackets in _L_ 2 (P) needed to cover the class _F_ with envelope function _F_ . That ~~_F_~~ ~~_**η**_~~ is
Lipschitz (and thus the bracketing integral is finite) follows from the fact that _ϕ_ is a sum
of products of Lipschitz functions and _D_ is bounded. We show this by showing that the
corresponding derivatives are all bounded, specifically

_∂_ _a_ _t_ _{_ 1 _−_ _π_ _t_ ( **h** _t_ ) _} −_ (1 _−_ _a_ _t_ ) _δπ_ _t_ ( **h** _t_ ) = _a_ _t_ _{_ 1 _−_ _π_ _t_ ( **h** _t_ ) _}_ _−_ (1 _−_ _a_ _t_ ) _π_ _t_ ( **h** _t_ ) _≤_
���� _∂δ_ � _δ/_ (1 _−_ _δ_ ) ����� ���� _δ_ [2] ����



_∂_ _a_ _t_ _{_ 1 _−_ _π_ _t_ ( **h** _t_ ) _} −_ (1 _−_ _a_ _t_ ) _δπ_ _t_ ( **h** _t_ ) = _a_ _t_ _{_ 1 _−_ _π_ _t_ ( **h** _t_ ) _}_ _−_ (1 _−_ _a_ _t_ ) _π_ _t_ ( **h** _t_ ) _≤_ 1 + 1 _/δ_ _ℓ_ 2
���� _∂δ_ � _δ/_ (1 _−_ _δ_ ) ����� ���� _δ_ [2] ����

_∂_ _δπ_ _t_ ( **h** _t_ ) _m_ _t_ ( **h** _t_ _,_ 1) + _{_ 1 _−_ _π_ _t_ ( **h** _t_ ) _}m_ _t_ ( **h** _t_ _,_ 0) _π_ _t_ ( **h** _t_ ) _{_ 1 _−_ _π_ _t_ ( **h** _t_ ) _}{m_ _t_ ( **h** _t_ _,_ 1) _−_ _m_ _t_ ( **h** _t_ _,_ 0) _}_

=

���� _∂δ_ � _δπ_ _t_ ( **h** _t_ ) + 1 _−_ _π_ _t_ ( **h** _t_ ) ����� ���� _{δπ_ _t_ ( **h** _t_ ) + 1 _−_ _π_ _t_ ( **h** _t_ ) _}_ [2] ����



_a_ _t_ _{_ 1 _−_ _π_ _t_ ( **h** _t_ ) _} −_ (1 _−_ _a_ _t_ ) _δπ_ _t_ ( **h** _t_ )
� _δ/_ (1 _−_ _δ_ )



_a_ _t_ _{_ 1 _−_ _π_ _t_ ( **h** _t_ ) _}_
=

_δ_ [2]

����� ����



_δ/_ (1 _−_ _δ_ )



����



_δπ_ _t_ ( **h** _t_ ) _m_ _t_ ( **h** _t_ _,_ 1) + _{_ 1 _−_ _π_ _t_ ( **h** _t_ ) _}m_ _t_ ( **h** _t_ _,_ 0)
� _δπ_ _t_ ( **h** _t_ ) + 1 _−_ _π_ _t_ ( **h** _t_ )



_π_ _t_ ( **h** _t_ ) _{_ 1 _−_ _π_ _t_ ( **h** _t_ ) _}{m_ _t_ ( **h** _t_ _,_ 1) _−_ _m_ _t_ ( **h** _t_ _,_ 0) _}_
=
����� ���� _{δπ_ _t_ ( **h** _t_ ) + 1 _−_ _π_ _t_ ( **h** _t_ ) _}_ [2]



_δπ_ _t_ ( **h** _t_ ) + 1 _−_ _π_ _t_ ( **h** _t_ )



_{δπ_ _t_ ( **h** _t_ ) + 1 _−_ _π_ _t_ ( **h** _t_ ) _}_ [2]



_≤|m_ _t_ ( **h** _t_ _,_ 1) _−_ _m_ _t_ ( **h** _t_ _,_ 0) _|/δ_ _ℓ_ [2]
_∂_ _δa_ _t_ + 1 _−_ _a_ _t_ _a_ _t_ _−_ _π_ _t_ ( **h** _t_ )

=

���� _∂δ_ � _δπ_ _t_ ( **h** _t_ ) + 1 _−_ _π_ _t_ ( **h** _t_ )����� ���� _{δπ_ _t_ ( **h** _t_ ) + 1 _−_ _π_ _t_ ( **h** _t_ ) _}_ [2]



_≤_ 1 _/δ_ _ℓ_ 2
����



_δa_ _t_ + 1 _−_ _a_ _t_
� _δπ_ _t_ ( **h** _t_ ) + 1 _−_ _π_ _t_ ( **h** _t_ )



_a_ _t_ _−_ _π_ _t_ ( **h** _t_ )
=
����� ���� _{δπ_ _t_ ( **h** _t_ ) + 1 _−_ _π_ _t_ ( **h** _t_ ) _}_ [2]



where we used the fact that, for all 0 _≤_ _π_ _t_ ( **h** _t_ ) _≤_ 1, we have


_{δπ_ _t_ ( **h** _t_ ) + 1 _−_ _π_ _t_ ( **h** _t_ ) _} ∈_ [ _δ ∧_ 1 _, δ ∨_ 1] _⊆_ [ _δ_ _ℓ_ _, δ_ _u_ ] _._


Therefore Ψ _n_ ( _·_ ) ⇝ G( _·_ ) since a function class with finite bracketing integral is necessarily
Donsker (e.g., Theorem 2.5.6 in van der Vaart & Wellner (1996)).


4


Now we consider the second statement, that _∥_ Ψ [�] _n_ _−_ Ψ _n_ _∥_ _D_ = _o_ P (1). First note that


_∥_ Ψ [�] _n_ _−_ Ψ _n_ _∥_ _D_ = _∥_ (Ψ [�] _n_ _−_ Ψ _n_ )( _σ/σ_ ˆ) + Ψ _n_ ( _σ −_ _σ_ ˆ) _/σ_ ˆ _∥_ _D_

_≤∥_ Ψ [�] _n_ _−_ Ψ _n_ _∥_ _D_ _∥σ/σ_ ˆ _∥_ _D_ + _∥σ/σ_ ˆ _−_ 1 _∥_ _D_ _∥_ Ψ _n_ _∥_ _D_

≲ _∥_ Ψ [�] _n_ _−_ Ψ _n_ _∥_ _D_ + _o_ P (1)


ˆ
where the last inequality follows since _∥σ/σ −_ 1 _∥_ _D_ = _o_ P (1) by Assumption 3 of Theorem 3,
and _∥_ Ψ _n_ _∥_ _D_ = _O_ P (1) follows from, e.g., Theorem 2.14.2 in van der Vaart & Wellner (1996),
since the function class _F_ _**η**_ has finite bracketing integral as shown above.


Now let _N_ = _n/K_ be the sample size in any group _k_ = 1 _, ..., K_, and denote the empirical
process over group _k_ units by G _[k]_ _n_ [=] _√N_ (P _[k]_ _n_ _[−]_ [P][). Then we have]


ˆ
Ψ� _n_ ( _δ_ ) _−_ Ψ _n_ ( _δ_ ) = _ψ_ ( _δ_ ) _−_ _ψ_ ( _δ_ ) _−_ G _n_ _{ϕ_ �( **Z** ; _**η**_ _, δ_ ) _}_

_σ_ ( _δ_ ) _/_ ~~_[√]_~~ ~~_n_~~


_[̸]_



_√_ ~~_n_~~ 1
=
_σ_ ( _δ_ ) _K_


_√_ ~~_n_~~
=
_Kσ_ ( _δ_ )


_[̸]_



_K_
�


_k_ =1


_K_
�


_k_ =1


_[̸]_



P _[k]_ _n_ _[{][ϕ]_ [(] **[Z]** [; ˆ] _**[η]**_ - _k_ _[, δ]_ [)] _[} −]_ _[ψ]_ [(] _[δ]_ [)] _[ −]_ [(][P] _[n]_ _[−]_ [P][)] _[ϕ]_ [(] **[Z]** [;] _**[ η]**_ _[, δ]_ [)]
� �


_[̸]_



1

G _[k]_ _n_ _ϕ_ ( **Z** ; ˆ _**η**_  - _k_ _, δ_ ) _−_ _ϕ_ ( **Z** ; _**η**_ _, δ_ ) + P _ϕ_ ( **Z** ; ˆ _**η**_  - _k_ _, δ_ ) _−_ _ϕ_ ( **Z** ; _**η**_ _, δ_ )

� ~~_√_~~ _N_ � � � ��


_[̸]_



� 1
~~_√_~~


_[̸]_



_≡_ _B_ _n,_ 1 ( _δ_ ) + _B_ _n,_ 2 ( _δ_ )


_[̸]_



where the first two equalities follow by definition, and the third by rearranging and noting
that _ψ_ ( _δ_ ) = P _{ϕ_ ( **Z** ; _**η**_ _, δ_ ) _}_ and [�] _k_ [P] _n_ _[k]_ _[{][ϕ]_ [(] **[Z]** [;] _**[ η]**_ _[, δ]_ [)] _[}]_ [ =][ �] _k_ [P] _[n]_ _[{][ϕ]_ [(] **[Z]** [;] _**[ η]**_ _[, δ]_ [)] _[}]_ [. Now we will an-]


_[̸]_



_k_ [P] _n_ _[k]_ _[{][ϕ]_ [(] **[Z]** [;] _**[ η]**_ _[, δ]_ [)] _[}]_ [ =][ �]


_[̸]_



that _ψ_ ( _δ_ ) = P _{ϕ_ ( **Z** ; _**η**_ _, δ_ ) _}_ and [�] _k_ [P] _n_ _[k]_ _[{][ϕ]_ [(] **[Z]** [;] _**[ η]**_ _[, δ]_ [)] _[}]_ [ =][ �] _k_ [P] _[n]_ _[{][ϕ]_ [(] **[Z]** [;] _**[ η]**_ _[, δ]_ [)] _[}]_ [. Now we will an-]

alyze the two pieces _B_ _n,_ 1 and _B_ _n,_ 2 in turn; showing that their supremum norms are both
_o_ P (1) completes the proof.


_[̸]_



For _B_ _n,_ 1, we have by the triangle inequality and since _K_ is fixed (independent of total
sample size _n_ ), that


_[̸]_



_K_
� G _[k]_ _n_ � _ϕ_ ( **Z** ; ˆ _**η**_ - _k_ _, δ_ ) _−_ _ϕ_ ( **Z** ; _**η**_ _, δ_ )� [�]

_k_ =1 ����


_[̸]_



_∥B_ _n,_ 1 _∥_ _D_ = sup
_δ∈D_


_[̸]_



1
����� ~~_√_~~ _Kσ_ ( _δ_ )


_[̸]_



≲ max sup _|_ G _n_ ( _f_ ) _|,_
_k_
_f_ _∈F_ _n_ _[k]_


where _F_ _n_ _[k]_ [=] _[ F]_ _**[η]**_ [ˆ] - _k_ _[−F]_ _**[η]**_ [for the function class] _[ F]_ _**[η]**_ [=] _[ {][ϕ]_ [(] _[·]_ [;] _**[ η]**_ _[, δ]_ [) :] _[ δ][ ∈D}]_ [ from before. Viewing]
the nuisance functions ˆ _**η**_ - _k_ as fixed given the training data **D** _[k]_ 0 [=] _[ {]_ **[Z]** _[i]_ [:] _[ S]_ _[i]_ _[̸]_ [=] _[ k][}]_ [, we can]
apply Theorem 2.14.2 in van der Vaart & Wellner (1996) to obtain




_[̸]_


�




_[̸]_


E




_[̸]_


sup _|_ G _n_ ( _f_ ) _|_ **D** _k_ 0
���

� _f_ _∈F_ _n_ _[k]_




_[̸]_


1
≲ _∥F_ _n_ _[k]_ _[∥]_
� 0




_[̸]_


�




_[̸]_


1 + log _N_ [ ] ( _ϵ∥F_ _n_ _[k]_ _[∥][,][ F]_ _n_ _[k]_ _[, L]_ [2] [(][P][))] _[ dϵ]_




_[̸]_


for envelope _F_ _n_ _[k]_ [. If we take] _[ F]_ _n_ _[ k]_ [(] **[z]** [) = sup] _δ∈D_ _[|][ϕ]_ [(] **[z]** [; ˆ] _**[η]**_ - _k_ _[, δ]_ [)] _[ −]_ _[ϕ]_ [(] **[z]** [;] _**[ η]**_ _[, δ]_ [)] _[|]_ [ then the first term]
_∥F_ _n_ _[k]_ _[∥]_ [in the product above is] _[ o]_ [P] [(1). Although the bracketing integral is finite for any fixed]
_**η**_, here the function class depends on _n_ through ˆ _**η**_ - _k_ so we need a more careful analysis.


5


Specifically, since _F_ _n_ _[k]_ [is Lipschitz, by Theorem 2.7.2 of van der Vaart & Wellner (1996)]
we have
1
log _N_ [ ] ( _ϵ∥F_ _n_ _[k]_ _[∥][,][ F]_ _n_ _[k]_ _[, L]_ [2] [(][P][))][ ≲]
_ϵ∥F_ _n_ _[k]_ _[∥][.]_


Therefore, letting _C_ _n_ _[k]_ [=] _[ ∥][F]_ _n_ _[ k]_ _[∥]_ [,]



1
1 + _dϵ_
_ϵC_ _n_ _[k]_

~~�~~



1
_∥F_ _n_ _[k]_ _[∥]_
� 0



1 + log _N_ [ ] ( _ϵ∥F_ _n_ _[k]_ _[∥][,][ F]_ _n_ _[k]_ _[, L]_ [2] [(][P][))] _[ dϵ]_ [ ≲] _[C]_ _n_ _[k]_

~~�~~



� 0 1



1 +

�



~~�~~



1 + 2 _C_ _n_ _[k]_

�



1 + [1]



_C_ _n_ _[k]_



= _C_ _n_ _[k]_



~~�~~



1 + [1]




[1] + 1 log

_C_ _n_ _[k]_ 2 _C_ _n_ _[k]_



~~�~~ �



1 +

�



�



��



= ~~�~~ _C_ _n_ _[k]_ [(] _[C]_ _n_ _[k]_ [+ 1) + (1] _[/]_ [2) log]



1 + 2 _C_ _n_ _[k]_

�



1 + [1]

_C_ _n_ _[k]_



which tends to zero as _C_ _n_ _[k]_ _[→]_ [0. Hence sup] _f_ _∈F_ _n_ _[k]_ _[|]_ [G] _[n]_ [(] _[f]_ [)] _[|]_ [ =] _[ o]_ [P] [(1) for each] _[ k]_ [, and since there]
are only finitely many splits _K_, we have


_∥B_ _n,_ 1 _∥_ _D_ = _o_ P (1) _._


To analyze _B_ _n,_ 2 ( _δ_ ) we require some new notation, and at first we typically suppress any
dependence on _δ_ for simplicity. Let _ψ_ (P; _Q_ ) denote the mean outcome under intervention
_Q_ for a population corresponding to observed data distribution P, and let _ϕ_ _[∗]_ ( **z** ; _**η**_ ) denote
its (centered) efficient influence function when _Q_ does not depend on P, as given in Lemma
2, which depends on nuisance functions _**η**_ = ( **m** _,_ _**π**_ ) = ( _m_ 0 _, m_ 1 _, ..., m_ _T_ _, π_ 1 _, π_ 2 _, ..., π_ _T_ ). Similarly let _ζ_ ( **z** ; _**η**_ ) denote the contribution to the efficient influence function _ϕ_ _[∗]_ ( **z** ; _**η**_ ) due to
estimating _Q_ when it depends on P, as given in Lemma 3. Then by definition


_ϕ_ ( **z** ; _**η**_ _, δ_ ) = _ϕ_ _[∗]_ ( **z** ; _**η**_ ) + _ψ_ (P; _Q_ ) + _ζ_ ( **z** ; _**η**_ ) _._


Hence, for any ~~_**η**_~~ we can write (1 _/_ _[√]_ ~~_n_~~ ~~)~~ _B_ _n,_ 2 ( _δ_ ) as


P _ϕ_ ( **Z** ; ~~_**η**_~~ ~~_,_~~ _δ_ ) _−_ _ϕ_ ( **Z** ; _**η**_ _, δ_ ) = _{ϕ_ _[∗]_ ( **z** ; ~~_**η**_~~ ~~)~~ + _ζ_ ( **z** ; ~~_**η**_~~ ~~)~~ + _ψ_ (P _, Q_ ) _} d_ P( **z** ) _−_ _ψ_ (P _, Q_ )
� � �


= _ϕ_ _[∗]_ ( **z** ; ~~_**η**_~~ ~~)~~ _d_ P( **z** ) + _ψ_ (P; _Q_ ) _−_ _ψ_ (P; _Q_ )
�


+ _ζ_ ( **z** ; ~~_**η**_~~ ~~)~~ _d_ P( **z** ) + _ψ_ (P; _Q_ ) _−_ _ψ_ (P; _Q_ )
�


where the first equality follows by definition and the second by rearranging.


In the following lemmas we analyze these two components of the remainder term _B_ _n,_ 2 ( _δ_ ).
Our results keep the intervention distribution _Q_ completely general, and so can be applied
to study other stochastic interventions, beyond those we focus on in this paper of the
incremental propensity score variety.


6


**Lemma 5.** _Let ψ_ (P; _Q_ ) _denote the mean outcome under intervention Q for a population_
_corresponding to observed data distribution_ P _, and let ϕ_ _[∗]_ ( **z** ; _**η**_ ) _denote its efficient influence_
_function when Q does not depend on_ P _, as given in Lemma 2, which depends on nuisance_
_functions_ _**η**_ = ( **m** _,_ _**π**_ ) = ( _m_ 0 _, m_ 1 _, ..., m_ _T_ _, π_ 1 _, π_ 2 _, ..., π_ _T_ ) _. Then for two distributions_ P _and_ P
_(the latter with corresponding nuisance functions_ ~~_**η**_~~ ~~_)_~~ _we have the expansion_


_ψ_ (P; _Q_ ) _−_ _ψ_ (P; _Q_ ) + _ϕ_ _[∗]_ ( **z** ; ~~_**η**_~~ ~~)~~ _d_ P( **z** )
�



�



_s−_ 1
�
� [�] _r_ =1



_t_
� _dQ_ _r_ _d_ P _r_
�� _r_ =1



=



_T_
�


_t_ =1



_t_
�


_s_ =1



_dπ_ _s_ _−_ ~~_dπ_~~ _s_
� ( _m_ _[∗]_ _t_ _[−]_ ~~_[m]_~~ _[t]_ [)] � ~~_dπ_~~ _s_



_dπ_ _r_

~~_dπ_~~ _r_



_where we define_


~~_m_~~ _t_ = ~~_m_~~ _t_ ( **H** _t_ _, A_ _t_ ) =
�



~~_m_~~ _t_ +1 _dQ_ _t_ +1 _d_ P _t_ +1 _,_ _m_ _[∗]_ _t_ [=]
�



~~_m_~~ _t_ +1 _dQ_ _t_ +1 _d_ P _t_ +1 _,_



_dQ_ _t_ = _dQ_ _t_ ( _A_ _t_ _|_ **H** _t_ ) _, dπ_ _t_ = _d_ P( _A_ _t_ _|_ **H** _t_ ) _, d_ P _t_ = _d_ P( **X** _t_ _|_ **H** _t−_ 1 _, A_ _t−_ 1 ) _._


_Proof._ First note that



_t_
�
� _s_ =0



_t_
�
�



�



E _{ϕ_ _[∗]_ ( **Z** ; ~~_**η**_~~ ~~)~~ _}_ = E


= E


= E



_T_
�


_t_ =0


_T_
�


_t_ =0



_T_
� ( _m_ _[∗]_ _t_ _[−]_ ~~_[m]_~~ _[t]_ [)]


_t_ =0



��



_dQ_ _s_
� ~~_dπ_~~ _s_ �



~~_m_~~ _t_ +1 _dQ_ _t_ +1 _−_ ~~_m_~~ _t_



_dQ_ _s_
� ~~_dπ_~~ _s_



���



_t_
�
� _s_ =0



_t_
�
�



�



~~_m_~~ _t_ +1 _dQ_ _t_ +1 _d_ P _t_ +1 _−_ ~~_m_~~ _t_



_dQ_ _s_
� ~~_dπ_~~ _s_



_t_
�


_s_ =0



_t_
�


_s_ =0



_dπ_ _s_ _d_ P _s_
�



=



_T_
� ( _m_ _[∗]_ _t_ _[−]_ ~~_[m]_~~ _[t]_ [)]

�

_t_ =0



_dQ_ _s_
� ~~_dπ_~~ _s_



where the first equality follows by definition, the second by iterated expectation (conditioning on ( **H** _t_ _, A_ _t_ ) and averaging over **X** _t_ +1 ), the third by definition of _m_ _[∗]_ _t_ [, and the fourth]
by repeated iterated expectation. Now we have



_T_
�


_t_ =0



_t_
�


_s_ =0



( _m_ _[∗]_ _t_ _[−]_ ~~_[m]_~~ _[t]_ [)]
�



_dQ_ _s_
� ~~_dπ_~~ _s_



_dπ_ _s_ _d_ P _s_
�



_dQ_ _t_ _d_ P _t_
�



_t−_ 1
�


_s_ =0



_dπ_ _s_ _dQ_ _s_ _d_ P _s_
� ~~_dπ_~~ _s_ �



=


=



_T_
�


_t_ =1



_T_
�


_t_ =1



_dπ_ _t_ _−_ ~~_dπ_~~ _t_
� ( _m_ _[∗]_ _t_ _[−]_ ~~_[m]_~~ _[t]_ [)] � ~~_dπ_~~ _t_



_dπ_ _r_
� ~~_dπ_~~ _r_



_t−_ 1
� _s_ =0 � ~~_dπ_~~ _dπ_ _ss_



_dQ_ _s_ _d_ P _s_ + ( _m_ _[∗]_ 0 _[−]_ ~~_[m]_~~ [0] [)]
�



+



_T_
�


_t_ =1



( _m_ _[∗]_ _t_ _[−]_ ~~_[m]_~~ _[t]_ [)] _[ dQ]_ _[t]_ _[d]_ [P] _[t]_
�



_t_
� ( _m_ _[∗]_ _t_ _[−]_ ~~_[m]_~~ _[t]_ [)]

�

_s_ =1



_t_
�� _r_ = _s_ _dQ_ _r_ _d_ P _r_ �� _dπ_ _s_ ~~_dπ_~~ _−_ _s_ ~~_dπ_~~ _s_


7



_s−_ 1
�
� _r_ =1



_dQ_ _r_ _d_ P _r_
�


_t_
� _dQ_ _s_ _d_ P _s_ + ( _m_ _[∗]_ 0 _[−]_ ~~_[m]_~~ [0] [)]


_s_ =1



+



_T_
�


_t_ =1



( _m_ _[∗]_ _t_ _[−]_ ~~_[m]_~~ _[t]_ [)]
�



_t_
�



where the first equality follows by adding and subtracting the second term in the sum (and
separating the _t_ = 0 term), and the second follows by repeating this process _t_ times (where
we use the convention that quantities at negative times like _dQ_ _−_ 1 are set to one). The last
terms in the last line above are a telescoping sum since



_T_
�


_t_ =1



_t_
� _dQ_ _s_ _d_ P _s_ =


_s_ =1



�



_t_
�



_t_
� _dQ_ _s_ _d_ P _s_ _−_

�
_s_ =1



( _m_ _[∗]_ _t_ _[−]_ ~~_[m]_~~ _[t]_ [)]
�



~~_m_~~ _t_ _dQ_ _t_ _d_ P _t_



_t−_ 1
� _dQ_ _s_ _d_ P _s_


_s_ =0



�



_t−_ 1
� _dQ_ _s_ _d_ P _s_


_s_ =0



=


=



_T_
�


_t_ =1


_T_
�


_t_ =1


_T_
�


_t_ =1



_m_ _[∗]_ _t_
��


_m_ _[∗]_ _t_
��



_m_ _[∗]_ _t_
�



_t_
� _dQ_ _s_ _d_ P _s_ _−_ _m_ _[∗]_ _t−_ 1

�
_s_ =1



_T_ _−_ 1
� _m_ _[∗]_ _t_

�

_t_ =1



_t_
� _dQ_ _s_ _d_ P _s_ _−_ _m_ _[∗]_ 0


_s_ =1



_t_
� _dQ_ _s_ _d_ P _s_ _−_


_s_ =1



= _m_ _[∗]_ _T_
�



_T_
� _dQ_ _s_ _d_ P _s_ _−_ _m_ _[∗]_ 0 [=] _[ m]_ [0] _[−]_ _[m]_ _[∗]_ 0 _[.]_


_s_ =1



Therefore the result follows after rearranging and noting _ψ_ _Q_ _[∗]_ [(][P][) =] _[ m]_ [0] [ and] _[ ψ]_ _Q_ _[∗]_ [(][P][) =] ~~_[m]_~~ [0] [.]


**Lemma 6.** _Using the same notation as in Lemma 5, let ζ_ ( **Z** ; _**η**_ ) _denote the contribution to_
_the efficient influence function ϕ_ _[∗]_ ( **Z** ; _**η**_ ) _as given in Lemma 3. Then for two intervention_
_distributions Q and Q (assumed to have densities dQ_ _t_ _and dQ_ _t_ _, respectively, for t_ = 1 _, ..., T_ _,_
_with respect to some dominating measure) we have the expansion_


_ψ_ (P; _Q_ ) _−_ _ψ_ (P; _Q_ ) + _ζ_ ( **z** ; ~~_**η**_~~ ~~)~~ _d_ P( **z** )
�



�



_t−_ 1
�
� _s_ =0



_d_ ~~_dπ_~~ _Q_ _ss_ _dπ_ _s_ _d_ P _s_



_dQ_ _s_



=



_T_
�


_t_ =1



( _φ_ _t_ _dπ_ _t_ ) ~~(~~ ~~_m_~~ _t_ _−_ _m_ _t_ ) _dν d_ P _t_
�



_s−_ 1
�
�� _r_ =0



_t−_ 1
� _dQ_ _r_ _d_ P _r_
� _r_ =0



_m_ _t_ _dν d_ P _t_
�



�



+


+



_T_
�


_t_ =1


_T_
�


_t_ =1



_t_
�


_s_ =1



_m_ _t_ ( _dQ_ _t_ _−_ _dQ_ _t_ + _φ_ _t_ _dπ_ _t_ ) _dν d_ P _t_
�



_dπ_ _s_ _−_ ~~_dπ_~~ _s_
( _φ_ _t_ _dπ_ _t_ )
� � ~~_dπ_~~ _s_



_dπ_ _s_

~~_dπ_~~ _s_



�



_t−_ 1
� _dQ_ _s_ _d_ P _s_
� _s_ =0



_t−_ 1
�
� _s_ =0



_Proof._ First note that


_ψ_ (P; _Q_ ) _−_ _ψ_ (P; _Q_ ) = _m_ _T_
�



_T_
� _dQ_ _t_ _d_ P _t_ _−_
� _t_ = _t_



_T_
� _dQ_ _t_ _d_ P _t_


_t_ = _t_



�



_T_ _−_ 1
� _dQ_ _t_ _d_ P _t_


_t_ =1



= _m_ _T_ ( _dQ_ _T_ _−_ _dQ_ _T_ ) _d_ P _T_
�


8



_T_ _−_ 1
� _dQ_ _t_ _d_ P _t_ + _m_ _T_ _dQ_ _T_ _d_ P _T_

�
_t_ =1


_T_ _−_ 1
� _dQ_ _t_ _d_ P _t_


_t_ =1



= _m_ _T_ ( _dQ_ _T_ _−_ _dQ_ _T_ ) _d_ P _T_
�



_T_ _−_ 1
� _dQ_ _t_ _d_ P _t_ + _m_ _T_ _−_ 1

�
_t_ =1



_t−_ 1
� _dQ_ _t_ _d_ P _t_


_s_ =0



=



_T_
�


_t_ =1



_m_ _t_ ( _dQ_ _t_ _−_ _dQ_ _t_ ) _d_ P _t_
�



where the first equality follows by definition, the second by adding and subtracting the last
term, the third by definition of _m_ _t_, and the fourth by repeating this process _T_ times.
Now we have that the expected contribution to the influence function due to estimating
_Q_ when it depends on P is



_t−_ 1
�
� _s_ =0



�
�



�



_T_
�


_t_ =1



�



_φ_ _t_ _dπ_ _t_ ~~_m_~~ _t_ _dν d_ P _t_



_t−_ 1
�
� _s_ =0



_d_ ~~_dπ_~~ _Q_ _ss_ _dπ_ _s_ _d_ P _s_



E



_T_
�


_t_ =1



_dQ_ _s_


~~_dπ_~~ _s_



_φ_ _t_ ~~_m_~~ _t_ _dν_ =



�



_t−_ 1
�
� _s_ =0



_d_ ~~_dπ_~~ _Q_ _ss_ _dπ_ _s_ _d_ P _s_



=


=



_T_
�


_t_ =1



_T_
�


_t_ =1



( _φ_ _t_ _dπ_ _t_ ) ~~(~~ ~~_m_~~ _t_ _−_ _m_ _t_ ) _dν d_ P _t_
�



( _φ_ _t_ _dπ_ _t_ ) ~~(~~ ~~_m_~~ _t_ _−_ _m_ _t_ ) _dν d_ P _t_
�



�



_t−_ 1
�
� _s_ =0



_d_ ~~_dπ_~~ _Q_ _ss_ _dπ_ _s_ _d_ P _s_



_dQ_ _s_



+



_T_
�


_t_ =1



( _φ_ _t_ _dπ_ _t_ ) _m_ _t_ _dν d_ P _t_
�



�



_t−_ 1
�
� _s_ =0



_d_ ~~_dπ_~~ _Q_ _ss_ _dπ_ _s_ _d_ P _s_



_dπ_ _s_ _−_ ~~_dπ_~~ _s_


~~_dπ_~~ _s_

��



_s−_ 1
�
� [�] _r_ =0



_t−_ 1
� _dQ_ _r_ _d_ P _r_
� _r_ =0



_t−_ 1
�
� _r_ =0



�



_dπ_ _s_

~~_dπ_~~ _s_



+


+



_T_
�


_t_ =1


_T_
�


_t_ =1



_t_
�


_s_ =1



( _φ_ _t_ _dπ_ _t_ ) _m_ _t_ _dν d_ P _t_
�



( _φ_ _t_ _dπ_ _t_ ) _m_ _t_ _dν d_ P _t_
�



�



_t−_ 1
� _dQ_ _s_ _d_ P _s_
� _s_ =0



where the first equality follows by iterated expectation, the second by adding and subtracting the second term in the sum, and the third by the same logic as in Lemma 5.
Now considering the last term in the above display plus _ψ_ (P; _Q_ ) _−_ _ψ_ (P; _Q_ ) we have



_t−_ 1
� _dQ_ _s_ _d_ P _s_
� _s_ =0 �



_ψ_ (P; _Q_ ) _−_ _ψ_ (P; _Q_ ) +


=


which yields the result.



_T_
�


_t_ =1


_T_
�


_t_ =1



( _φ_ _t_ _dπ_ _t_ ) _m_ _t_ _dν d_ P _t_
�



�



_m_ _t_ ( _dQ_ _t_ _−_ _dQ_ _t_ + _φ_ _t_ _dπ_ _t_ ) _dν d_ P _t_
�



_t−_ 1
� _dQ_ _s_ _d_ P _s_
� _s_ =0



Now we need to translate the remainder terms from Lemmas 5 and 6 to the incremental

propensity score intervention setting. The remainder from Lemma 5 equals



�



_T_
�


_t_ =1



_t_
�


_s_ =1



_dπ_ _s_ _−_ ~~_dπ_~~ _s_
� ( _m_ _[∗]_ _t_ _[−]_ ~~_[m]_~~ _[t]_ [)] � ~~_dπ_~~ _s_



_s−_ 1
�
� [�] _r_ =1


9



_dπ_ _r_

~~_dπ_~~ _r_



_t_
� _dQ_ _r_ _d_ P _r_
�� _r_ =1



_t_
�
�� _r_ =1


_t_
�


_s_ =1



=


≲



_T_
�


_t_ =1



_T_
�


_t_ =1



~~(~~ ~~_m_~~ _t_ +1 _−_ _m_ _t_ +1 ) _dQ_ _t_ +1 _d_ P _t_ +1 + _m_ _t_ +1 ( _dQ_ _t_ +1 _−_ _dQ_ _t_ +1 ) _d_ P _t_ +1
��



~~_∥m_~~ _t_ +1 _−_ _m_ _t_ +1 _∥_ + ~~_∥π_~~ _t_ +1 _−_ _π_ _t_ +1 _∥_ + _∥m_ _t_ _−_ ~~_m_~~ _t_ _∥_ _∥π_ _s_ _−_ ~~_π_~~ _s_ _∥_
� �



_dπ_ _s_ _−_ ~~_dπ_~~ _s_
+ ( _m_ _t_ _−_ ~~_m_~~ _t_ )
� [�] ~~_dπ_~~ _s_



_s−_ 1
�
� [�] _r_ =1



�



_dπ_ _r_ _d_ P _r_
�



_dπ_ _r_

~~_dπ_~~ _r_



_t_
�
�� _r_ =1



_dQ_ _r_
� _dπ_ _r_



_t_
�


_s_ =1



where the last inequality follows since


_δ_ (2 _a_ _t_ _−_ 1) ~~(~~ ~~_π_~~ _t_ _−_ _π_ _t_ )
_dQ_ _t_ _−_ _dQ_ _t_ =
( _δ_ ~~_π_~~ _t_ + 1 _−_ _π_ _t_ )( _δπ_ _t_ + 1 _−_ _π_ _t_ ) _[.]_



For the remainder from Lemma 6 first note that


_φ_ _t_ _dπ_ _t_ = _[δ]_ [(][2] _[a]_ _[t]_ _[ −]_ [1]

� ( _δ_ ~~_π_~~ _t_ + 1



_φ_ _t_ _dπ_ _t_ = _[δ]_ [(][2] _[a]_ _[t]_ _[ −]_ [1][)(] _[π]_ _[t]_ _[ −]_ ~~_[π]_~~ _[t]_ [)]



( _δ_ ~~_π_~~ _t_ + 1 _−_ ~~_π_~~ _t_ ) [2]



where we used the form of the efficient influence function derived in Lemma 4. Combining
the two previous expressions gives



_dQ_ _t_ _−_ _dQ_ _t_ +
�



_δ_ ( _δ −_ 1)(2 _a_ _t_ _−_ 1) ~~(~~ ~~_π_~~ _t_ _−_ _π_ _t_ ) [2]
_φ_ _t_ _dπ_ _t_ =

( _δ_ ~~_π_~~ _t_ + 1 _−_ ~~_π_~~ _t_ ) [2] ( _δπ_ _t_ + 1 _−_ _π_ _t_ ) _[.]_



Thus the remainder from Lemma 6 is



_T_
�


_t_ =1



�



( _φ_ _t_ _dπ_ _t_ ) ~~(~~ ~~_m_~~ _t_ _−_ _m_ _t_ ) _dν d_ P _t_
�



_t−_ 1
�
� _s_ =0



_d_ ~~_dπ_~~ _Q_ _ss_ _dπ_ _s_ _d_ P _s_



_s−_ 1
�
�� _r_ =0



_t−_ 1
� _dQ_ _r_ _d_ P _r_
� _r_ =0



�



+


+



_T_
�


_t_ =1


_T_
�


_t_ =1



_t_

_dπ_ _s_ _−_ ~~_dπ_~~ _s_

� _s_ =1 � ( _φ_ _t_ _dπ_ _t_ ) � ~~_dπ_~~ _s_



_m_ _t_ ( _dQ_ _t_ _−_ _dQ_ _t_ + _φ_ _t_ _dπ_ _t_ ) _dν d_ P _t_
�



_m_ _t_ _dν d_ P _t_
�



_dπ_ _s_

~~_dπ_~~ _s_



_t−_ 1
� _dQ_ _s_ _d_ P _s_
� _s_ =0



�


_._



�



≲



_T_
� _∥π_ _t_ _−_ ~~_π_~~ _t_ _∥_ ~~_∥m_~~ _t_ _−_ _m_ _t_ _∥_ +

_t_ =1 �



_t_
� _∥π_ _s_ _−_ ~~_π_~~ _s_ _∥_ + _∥π_ _t_ _−_ ~~_π_~~ _t_ _∥_


_s_ =1



The condition given in Theorem 3, that for _s ≤_ _t ≤_ _T_ we have

ˆ ˆ ˆ
sup _∥m_ _t,δ_ _−_ _m_ _t,δ_ _∥_ + _∥π_ _t_ _−_ _π_ _t_ _∥_ _∥π_ _s_ _−_ _π_ _s_ _∥_ = _o_ P (1 _/_ _[√]_ _n_ ) _,_
� _δ∈D_ �


therefore ensures that the above remainders from Lemmas 5 and 6 are negligible up to
order _n_ _[−]_ [1] _[/]_ [2] uniformly in _δ_ . Therefore _∥B_ _n,_ 2 _∥_ _D_ = _o_ P (1), which concludes the proof.


10


#### **8.5 Proof of Theorem 4**

As in the proof of Theorem 3, let _∥f_ _∥_ _D_ = sup _δ∈D_ _|f_ ( _δ_ ) _|_ denote the supremum norm with
respect to _δ_, and define the processes


�
Ψ _n_ ( _δ_ ) = _[√]_ _n{ψ_ ˆ( _δ_ ) _−_ _ψ_ ( _δ_ ) _}/σ_ ˆ( _δ_ )

�
Ψ _[∗]_ _n_ [(] _[δ]_ [) =][ G] _[n]_ [[] _[ξ][{][ϕ]_ [(] **[Z]** [; ˆ] _**[η]**_        - _S_ _[, δ]_ [)] _[ −]_ _[ψ]_ [ˆ][(] _[δ]_ [)] _[}][/][σ]_ [ˆ][(] _[δ]_ [)]]

Ψ _[∗]_
_n_ [(] _[δ]_ [) =][ G] _[n]_ [[] _[ξ][{][ϕ]_ [(] **[Z]** [;] _**[ η]**_ _[, δ]_ [)] _[ −]_ _[ψ]_ [(] _[δ]_ [)] _[}][/σ]_ [(] _[δ]_ [)]] _[.]_


Note that the star superscripts denote multiplier bootstrap processes. As before, let G( _δ_ )
denote the mean-zero Gaussian process with covariance E _{ϕ_ �( **Z** ; _**η**_ _, δ_ 1 ) _ϕ_ �( **Z** ; _**η**_ _, δ_ 2 ) _}_ .
Since


ˆ
P _ψ_ ( _δ_ ) _−_ _[c]_ [ˆ] _[α]_ _[σ]_ [ˆ][(] _[δ]_ [)] _≤_ _ψ_ ( _δ_ ) _≤_ _ψ_ [ˆ] ( _δ_ ) + _[c]_ [ˆ] _[α]_ _[σ]_ [ˆ][(] _[δ]_ [)] _,_ for all _δ ∈D_
� ~~_√n_~~ ~~_√n_~~ �



�����



ˆ
_≤_ _c_ _α_
�����



�



= P



sup
_δ∈D_

�



ˆ
_ψ_ ( _δ_ ) _−_ _ψ_ ( _δ_ )


ˆ
_σ_ ( _δ_ ) _/_ ~~_[√]_~~ ~~_n_~~



ˆ
= P _∥_ Ψ [�] _n_ _∥_ _D_ _≤_ _c_ _α_ _,_
� �



the result of Theorem 4 requires that we show

ˆ
P _∥_ Ψ [�] _n_ _∥_ _D_ _≤_ _c_ _α_ _−_ P _∥_ Ψ [�] _[∗]_ _n_ _[∥]_ _[D]_ _[≤]_ _[c]_ [ˆ] _[α]_ = _o_ (1) _,_
��� � � � ����


which yields the desired result since P( _∥_ Ψ [�] _[∗]_ _n_ _[∥]_ _[D]_ _[≤]_ _[c]_ [ˆ] _[α]_ [) = 1] _[ −]_ _[α]_ [ by definition of ˆ] _[c]_ _[α]_ [.]
We showed in the proof of Theorem 3 that _∥_ Ψ [�] _n_ _−_ Ψ _n_ _∥_ _D_ = _o_ P (1) which implies that
_|∥_ Ψ [�] _n_ _∥_ _D_ _−∥_ Ψ _n_ _∥_ _D_ _|_ = _o_ P (1), and by Corollary 2.2 in Chernozhukov _et al._ (2014) we have

_∥_ Ψ _n_ _∥_ _D_ _−∥_ G _∥_ _D_ = _o_ P (1) _._
��� ���


Hence by Lemma 2.3 in Chernozhukov _et al._ (2014) it follows that



sup
_t∈_ R



P _∥_ Ψ [�] _n_ _∥_ _D_ _≤_ _t_ _−_ P _∥_ G _∥_ _D_ _≤_ _t_ = _o_ (1) _._
��� � � � ����



Similarly, by Corollary 2.2 of Belloni _et al._ (2015) we have

�
_∗_ _∗_
_∥_ Ψ _n_ _[∥]_ _[D]_ _[−∥]_ [Ψ] _[∗]_ _n_ _[∥]_ _[D]_ = _o_ P (1) _,_ _∥_ Ψ _n_ _[∥]_ _[D]_ _[−∥]_ [G] _[∥]_ _[D]_ = _o_ P (1)
��� ��� ��� ���


so that again by Lemma 2.3 in Chernozhukov _et al._ (2014)



sup
_t∈_ R



P _∥_ Ψ [�] _[∗]_ _n_ _[∥]_ _[D]_ _[≤]_ _[t]_ _−_ P _∥_ G _∥_ _D_ _≤_ _t_ = _o_ (1) _._
��� � � � ����



This yields the result, since _|_ P( _∥_ Ψ [�] _n_ _∥_ _D_ _≤_ _c_ ˆ _α_ ) _−_ P( _∥_ Ψ [�] _[∗]_ _n_ _[∥]_ _[D]_ _[≤]_ _[c]_ [ˆ] _[α]_ [)] _[|]_ [ is bounded above by]



P _∥_ Ψ [�] _[∗]_ _n_ _[∥]_ _[D]_ _[≤]_ _[t]_ _−_ P _∥_ G _∥_ _D_ _≤_ _t_ = _o_ (1) _._
��� � � � ����



sup
_t∈_ R



P _∥_ Ψ [�] _n_ _∥_ _D_ _≤_ _t_ _−_ P _∥_ G _∥_ _D_ _≤_ _t_ + sup
��� � � � ���� _t∈_ R


11


#### **8.6 R Code**

```
### this function requires the following inputs:
### dat: dataframe (in long not wide form if longitudinal) with columns
### ‘time’, ‘id’, outcome ‘y’, treatment ‘a’
### x.trt: covariate matrix for treatment regression
### x.out: covariate matrix for outcome regression
### delta.seq: sequence of delta values
### nsplits: number of sample splits
### NOTE: dat, x.trt, x.out should all have the same number of rows

ipsi <- function(dat, x.trt, x.out, delta.seq, nsplits){

# setup storage
ntimes <- length(table(dat$time)); n <- length(unique(dat$id))
k <- length(delta.seq); ifvals <- matrix(nrow=n,ncol=k); est.eff <- rep(NA,k)
wt <- matrix(nrow=n*ntimes,ncol=k); cumwt <- matrix(nrow=n*ntimes,ncol=k)
rt <- matrix(nrow=n*ntimes,ncol=k); vt <- matrix(nrow=n*ntimes,ncol=k)

s <- sample(rep(1:nsplits,ceiling(n/nsplits))[1:n])
slong <- rep(s,rep(ntimes,n))

for (split in 1:nsplits){ print(paste("split",split)); flush.console()

# fit treatment model

trtmod <- ranger(a ~ ., dat=cbind(x.trt,a=dat$a)[slong!=split,])
dat$ps <- predict(trtmod, data=x.trt)$predictions

for (j in 1:k){ print(paste("delta",j)); flush.console()
delta <- delta.seq[j]

# compute weights
wt[,j] <- (delta*dat$a + 1-dat$a)/(delta*dat$ps + 1-dat$ps)
cumwt[,j] <- as.numeric(t(aggregate(wt[,j],by=list(dat$id),cumprod)[,-1]))
vt[,j] <- (1-delta)*(dat$a*(1-dat$ps) - (1-dat$a)*delta*dat$ps)/delta

# fit outcome models

outmod <- vector("list",ntimes); rtp1 <- dat$y[dat$time==end]
print("fitting regressions"); flush.console()
for (i in 1:ntimes){
 t <- rev(unique(dat$time))[i]
 outmod[[i]] <- ranger(rtp1 ~ .,
  dat=cbind(x.out,rtp1)[dat$time==t & slong!=split,])
 newx1 <- x.out[dat$time==t,]; newx1$a <- 1

```

12


```
 m1 <- predict(outmod[[i]], data=newx1)$predictions
 newx0 <- x.out[dat$time==t,]; newx0$a <- 0
 m0 <- predict(outmod[[i]], data=newx0)$predictions
 pi.t <- dat$ps[dat$time==t]
 rtp1 <- (delta*pi.t*m1 + (1-pi.t)*m0) / (delta*pi.t + 1-pi.t)
 rt[dat$time==t,j] <- rtp1 }

ifvals[s==split,j] <- ((cumwt[,j]*dat$y)[dat$time==end] +
 aggregate(cumwt[,j]*vt[,j]*rt[,j],by=list(dat$id),sum)[,-1])[s==split]

} }

# compute estimator
for (j in 1:k){ est.eff[j] <- mean(ifvals[,j]) }

# compute asymptotic variance
sigma <- sqrt(apply(ifvals,2,var))
eff.ll <- est.eff-1.96*sigma/sqrt(n); eff.ul <- est.eff+1.96*sigma/sqrt(n)

# multiplier bootstrap
eff.mat <- matrix(rep(est.eff,n),nrow=n,byrow=T)
sig.mat <- matrix(rep(sigma,n),nrow=n,byrow=T)
ifvals2 <- (ifvals-eff.mat)/sig.mat
nbs <- 10000; mult <- matrix(2*rbinom(n*nbs,1,.5)-1,nrow=n,ncol=nbs)
maxvals <- sapply(1:nbs, function(col){
 max(abs(apply(mult[,col]*ifvals2,2,sum)/sqrt(n))) } )
calpha <- quantile(maxvals, 0.95)
eff.ll2 <- est.eff-calpha*sigma/sqrt(n); eff.ul2 <- est.eff+calpha*sigma/sqrt(n)

return(list(est=est.eff, sigma=sigma, ll1=eff.ll,ul1=eff.ul,
 calpha=calpha, ll2=eff.ll2,ul2=eff.ul2))

}

```

13


