## **Reducing Gender Bias in Neural Machine Translation as a Domain** **Adaptation Problem**

**Danielle Saunders** and **Bill Byrne**
Department of Engineering, University of Cambridge, UK
_{_ ds636, wjb31 _}_ @cam.ac.uk



**Abstract**


Training data for NLP tasks often exhibits gender bias in that fewer sentences refer to women

than to men. In Neural Machine Translation

(NMT) gender bias has been shown to reduce
translation quality, particularly when the target
language has grammatical gender. The recent
WinoMT challenge set allows us to measure
this effect directly (Stanovsky et al., 2019).


Ideally we would reduce system bias by simply debiasing all data prior to training, but
achieving this effectively is itself a challenge.
Rather than attempt to create a ‘balanced’
dataset, we use transfer learning on a small
set of trusted, gender-balanced examples. This
approach gives strong and consistent improvements in gender debiasing with much less computational cost than training from scratch.


A known pitfall of transfer learning on new domains is ‘catastrophic forgetting’, which we
address both in adaptation and in inference.
During adaptation we show that Elastic Weight
Consolidation allows a performance trade-off
between general translation quality and bias reduction. During inference we propose a latticerescoring scheme which outperforms all systems evaluated in Stanovsky et al. (2019) on
WinoMT with no degradation of general test
set BLEU, and we show this scheme can be
applied to remove gender bias in the output
of ‘black box‘ online commercial MT systems.
We demonstrate our approach translating from
English into three languages with varied linguistic properties and data availability.


**1** **Introduction**


As language processing tools become more prevalent concern has grown over their susceptibility to
social biases and their potential to propagate bias
(Hovy and Spruit, 2016; Sun et al., 2019). Natural language training data inevitably reflects biases
present in our society. For example, gender bias



manifests itself in training data which features more
examples of men than of women. Tools trained on
such data will then exhibit or amplify the biases
(Zhao et al., 2017) and their harmful stereotypes.
Gender bias is a particularly important problem
for Neural Machine Translation (NMT) into genderinflected languages. An over-prevalence of some
gendered forms in the training data leads to translations with identifiable errors (Stanovsky et al.,
2019). Translations are better for sentences involving men and for sentences containing stereotypical
gender roles. For example, mentions of male doctors are more reliably translated than those of male
nurses (Sun et al., 2019; Prates et al., 2019).
Recent approaches to the bias problem in NLP
have involved training from scratch on artificially
gender-balanced versions of the original dataset
(Zhao et al., 2018; Zmigrod et al., 2019) or with debiased embeddings (Escude Font and Costa-juss ´ a `,
2019; Bolukbasi et al., 2016). While these approaches may be effective, training from scratch
is inefficient and gender-balancing embeddings or
large parallel datasets are challenging problems
(Gonen and Goldberg, 2019).
Instead we propose treating gender debiasing
as a domain adaptation problem, since NMT models can very quickly adapt to a new domain (Freitag and Al-Onaizan, 2016). To the best of our
knowledge this work is the first to attempt NMT
bias reduction by fine-tuning, rather than retraining. We consider three aspects of this adaptation
problem: creating less biased adaptation data, parameter adaptation using this data, and inference
with the debiased models produced by adaptation.
Regarding data, we suggest that a small, trusted
gender-balanced set could allow more efficient and
effective gender debiasing than a larger, noisier
set. To explore this we create a tiny, handcrafted
profession-based dataset for transfer learning. For
contrast, we also consider fine-tuning on a coun

terfactual subset of the full dataset and propose
a straightforward scheme for artificially genderbalancing parallel text for NMT.

We find that during domain adaptation improvement on the gender-debiased domain comes at the
expense of translation quality due to catastrophic
forgetting (French, 1999). We can balance improvement and forgetting with a regularised training procedure, Elastic Weight Consolidation (EWC), or in
inference by a two-step lattice rescoring procedure.

We experiment with three language pairs, assessing the impact of debiasing on general domain BLEU and on the WinoMT challenge set
(Stanovsky et al., 2019). We find that continued
training on the handcrafted set gives far stronger
and more consistent improvements in genderdebiasing with orders of magnitude less training
time, although as expected general translation performance as measured by BLEU decreases.

We further show that regularised adaptation with
EWC can reduce bias while limiting degradation in
general translation quality. We also present a lattice
rescoring procedure in which initial hypotheses produced by the biased baseline system are transduced
to create gender-inflected search spaces which can
be rescored by the adapted model. We believe
this approach, rescoring with models targeted to
remove bias, is novel in NMT. The rescoring procedure improves WinoMT accuracy by up to 30%
with no decrease in BLEU on the general test set.

Recent recommendations for ethics in Artificial
Intelligence have suggested that social biases or
imbalances in a dataset be addressed prior to model
training (HLEG, 2019). This recommendation presupposes that the source of bias in a dataset is both
obvious and easily adjusted. We show that debiasing a full NMT dataset is difficult, and suggest
alternative efficient and effective approaches for
debiasing a model after it is trained. This avoids
the need to identify and remove all possible biases
prior to training, and has the added benefit of preserving privacy, since no access to the original data
or knowledge of its contents is required. As evidence, in section 3.4.5, we show this scheme can
be applied to remove gender bias in the output of
black box online commercial MT systems.


**1.1** **Related work**


Vanmassenhove et al. (2018) treat gender as a domain for machine translation, training from scratch
by augmenting Europarl data with a tag indicat


ing the speaker’s gender. This does not inherently remove gender bias from the system but allows control over the translation hypothesis gender.
Moryossef et al. (2019) similarly prepend a short
phrase at inference time which acts as a gender
domain label for the entire sentence. These approaches are not directly applicable to text which
may have more than one gendered entity per sentence, as in coreference resolution tasks.


Escude Font and Costa-juss ´ a ` (2019) train NMT
models from scratch with debiased word embed
dings. They demonstrate improved performance on
an English-Spanish occupations task with a single
profession and pronoun per sentence. We assess
our fine-tuning approaches on the WinoMT coreference set, with two entities to resolve per sentence.
For monolingual NLP tasks a typical approach
is gender debiasing using counterfactual data augmentation where for each gendered sentence in the
data a gender-swapped equivalent is added. Zhao
et al. (2018) show improvement in coreference resolution for English using counterfactual data. Zmigrod et al. (2019) demonstrate a more complicated
scheme for gender-inflected languages. However,
their system focuses on words in isolation, and is
difficult to apply to co-reference and conjunction
situations with more than one term to swap, reducing its practicality for large MT datasets.
Recent work recognizes that NMT can be
adapted to domains with desired attributes using
small datasets (Farajian et al., 2017; Michel and
Neubig, 2018). Our choice of a small, trusted
dataset for adaptation specifically to a debiased
domain connects also to recent work in data selec
tion by Wang et al. (2018), in which fine-tuning
on less noisy data reduces translation noise. Similarly we propose fine-tuning on less biased data to
reduce gender bias in translations. This is loosely
the inverse of the approach described by Park et al.
(2018) for monolingual abusive language detection,
which pre-trains on a larger, less biased set.


**2** **Gender bias in machine translation**


We focus on translating coreference sentences containing professions as a representative subset of the
gender bias problem. This follows much recent
work on NLP gender bias (Rudinger et al., 2018;
Zhao et al., 2018; Zmigrod et al., 2019) including
the release of WinoMT, a relevant challenge set for
NMT (Stanovsky et al., 2019).
A sentence that highlights gender bias is:


_The_ _**doctor**_ _told the nurse that_ _**she**_ _had been busy._
A human translator carrying out coreference resolution would infer that ‘she’ refers to the doctor,
and correctly translate the entity to German as _Die_
_Arztin_ _¨_ . An NMT model trained on a biased dataset

in which most doctors are male might incorrectly
default to the masculine form, _Der Arzt_ .

Data bias does not just affect translations of the
stereotyped roles. Since NMT inference is usually
left-to-right, a mistranslation can lead to further,
more obvious mistakes later in the translation. For

example, our baseline en-de system translates the
English sentence
_The cleaner hates the_ _**developer**_ _because_ _**she**_ _al-_
_ways leaves the room dirty._

to the German

_Der Reiniger haßt_ _**den Entwickler**_ _, weil_ _**er**_ _den_
_Raum immer schmutzig l¨asst._
Here not only is ‘developer’ mistranslated as
the masculine _den Entwickler_ instead of the fem
inine _die Entwicklerin_, but an unambiguous pronoun translation later in the sentence is incorrect:

_er_ (‘he’) is produced instead of _sie_ (‘she’).
In practice, not all translations with genderinflected words can be unambiguously resolved.
A simple example is:
_The doctor had been busy._
This would likely be translated with a masculine
entity according to the conventions of a language,
unless extra-sentential context was available. As

well, some languages have adopted gender-neutral
singular pronouns and profession terms, both to
include non-binary people and to avoid the social biases of gendered language (Misersky et al.,
2019). However, the target languages supported
by WinoMT lack widely-accepted non-binary inflection conventions (Ackerman, 2019). This paper
addresses gender bias that can be resolved at the
sentence level and evaluated with existing test sets,
and does not address these broader challenges.


**2.1** **WinoMT challenge set and metrics**


WinoMT (Stanovsky et al., 2019) is a recently proposed challenge set for gender bias in NMT. Moreover it is the only significant challenge set we are
aware of to evaluate translation gender bias comparably across several language pairs. It permits automatic bias evaluation for translation from English
to eight target languages with grammatical gender.
The source side of WinoMT is 3888 concatenated

sentences from Winogender (Rudinger et al., 2018)



and WinoBias (Zhao et al., 2018). These are coref
erence resolution datasets in which each sentence

contains a primary entity which is co-referent with
a pronoun – _the doctor_ in the first example above
and _the developer_ in the second – and a secondary
entity – _the nurse_ and _the cleaner_ respectively.
WinoMT evaluation extracts the grammatical
gender of the primary entity from each translation
hypothesis by automatic word alignment followed
by morphological analysis. WinoMT then compares the translated primary entity with the gold
gender, with the objective being a correctly gendered translation. The authors emphasise the following metrics over the challenge set:


_•_ **Accuracy** – percentage of hypotheses with
the correctly gendered primary entity.


_•_ **∆G** – difference in _F_ 1 score between the set

of sentences with masculine entities and the

set with feminine entities.


_•_ **∆S** – difference in accuracy between the set
of sentences with pro-stereotypical (‘pro’) entities and those with anti-stereotypical (‘anti’)
entities, as determined by Zhao et al. (2018)
using US labour statistics. For example, the
‘pro’ set contains male doctors and female
nurses, while ‘anti’ contains female doctors

and male nurses.


Our main objective is increasing accuracy. We
also report on ∆ _G_ and ∆ _S_ for ease of comparison
to previous work. Ideally the absolute values of
∆ _G_ and ∆ _S_ should be close to 0. A high positive
∆ _G_ indicates that a model translates male entities

better, while a high positive ∆ _S_ indicates that a
model stereotypes male and female entities. Large
negative values for ∆ _G_ and ∆ _S_, indicating a bias
towards female or anti-stereotypical translation, are
as undesirable as large positive values.
We note that ∆ _S_ can be significantly skewed by
low-accuracy systems. A model generating male
forms for most test sentences, stereotypical roles
or not, will have very low ∆ _S_, since its pro- and
anti-stereotypical class accuracy will both be about
50%. Consequently in Appendix A we report:


_•_ **M:F** – ratio of hypotheses with male predictions to those with female predictions.


This should be close to 1.0, since WinoMT bal
ances male- and female-labelled sentences. M:F

correlates strongly with ∆ _G_, but we consider M:F


easier to interpret, particularly since very high or
low M:F reduce the relevance of ∆ _S_ .

Finally, we wish to reduce gender bias without reducing translation performance. We report
**BLEU** (Papineni et al., 2002) on separate, general
test sets for each language pair. WinoMT is designed to work without target language references,
and so it is not possible to measure translation performance on this set by measures such as BLEU.


**2.2** **Gender debiased datasets**


**2.2.1** **Handcrafted profession dataset**

Our hypothesis is that the absence of gender bias
can be treated as a small domain for the purposes of
NMT model adaptation. In this case a well-formed
small dataset may give better results than attempts
at debiasing the entire original dataset.
We therefore construct a tiny, trivial set of
gender-balanced English sentences which we can
easily translate into each target language. The sentences follow the template:


_The_ [ _PROFESSION_ ] _finished_ [ _his|her_ ] _work._

We refer to this as the _handcrafted_ set [1] . Each profession is from the list collected by Prates et al.
(2019) from US labour statistics. We simplify this
list by removing field-specific adjectives. For example, we have a single profession ‘engineer’, as opposed to specifying industrial engineer, locomotive
engineer, etc. In total we select 194 professions,
giving just 388 sentences in a gender-balanced set.
With manually translated masculine and feminine templates, we simply translate the masculine and feminine forms of each listed profession
for each target language. In practice this translation is via an MT first-pass for speed, followed by
manual checking, but given available lexicons this
could be further automated. We note that the hand
crafted sets contain no examples of coreference
resolution and very little variety in terms of grammatical gender. A set of more complex sentences
targeted at the coreference task might further improve WinoMT scores, but would be more difficult
to produce for new languages.
We wish to distinguish between a model which
improves gender translation, and one which improves its WinoMT scores simply by learning the
vocabulary for previously unseen or uncommon
professions. We therefore create a _handcrafted no-_
_overlap_ set, removing source sentences with profes

1 Handcrafted sets available at [https://github.](https://github.com/DCSaunders/gender-debias)
[com/DCSaunders/gender-debias](https://github.com/DCSaunders/gender-debias)



sions occurring in WinoMT to leave 216 sentences.
We increase this set back to 388 examples with
balanced adjective-based sentences in the same pattern, e.g. _The tall_ [ _man_ _|_ _woman_ ] _finished_ [ _his_ _|_ _her_ ]
_work_ .


**2.2.2** **Counterfactual datasets**


Figure 1: Generating counterfactual datasets for adaptation. The **Original** set is 1 _||_ 2, a simple subset of the
full dataset. **FTrans original** is 1 _||_ 3, **FTrans swapped**
is 4 _||_ 5, and **Balanced** is 1,4 _||_ 2,5


For contrast, we fine-tune on an approximated
counterfactual dataset. Counterfactual data augmentation is an intuitive solution to bias from data

over-representation (Lu et al., 2018). It involves
identifying the subset of sentences containing bias –
in this case gendered terms – and, for each one,
adding an equivalent sentence with the bias reversed – in this case a gender-swapped version.

While counterfactual data augmentation is relatively simple for sentences in English, the process
for inflected languages is challenging, involving
identifying and updating words that are co-referent
with all gendered entities in a sentence. Genderswapping MT training data additionally requires
that the same entities are swapped in the corresponding parallel sentence. A robust scheme for
gender-swapping multiple entities in inflected language sentences directly, together with corresponding parallel text, is beyond the scope of this paper.
Instead we suggest a rough but straightforward approach for counterfactual data augmentation for
NMT which to the best of our knowledge is the
first application to parallel sentences.

We first perform simple gender-swapping on the
subset of the English source sentences with gendered terms. We use the approach described in
Zhao et al. (2018) which swaps a fixed list of gen

dered stopwords (e.g. _man_ / _woman_, _he_ / _she_ ). [2] . We
then greedily forward-translate the gender-swapped
English sentences with a baseline NMT model
trained on the the full source and target text, producing gender-swapped target language sentences.
This lets us compare four related sets for gender
debiasing adaptation, as illustrated in Figure 1:


_•_ **Original** : a subset of parallel sentences from
the original training data where the source
sentence contains gendered stopwords.


_•_ **Forward-translated (FTrans) original** : the
source side of the _original_ set with forwardtranslated target sentences.


_•_ **Forward-translated** **(FTrans)** **swapped** :
the _original_ source sentences are genderswapped, then forward-translated to produce
gender-swapped target sentences.


_•_ **Balanced** : the concatenation of the _original_
and _FTrans swapped_ parallel datasets. This is
twice the size of the other counterfactual sets.


Comparing performance in adaptation of _FTrans_
_swapped_ and _FTrans original_ lets us distinguish
between the effects of gender-swapping and of obtaining target sentences from forward-translation.


**2.3** **Debiasing while maintaining general**
**translation performance**


Fine-tuning a converged neural network on data
from a distinct domain typically leads to catastrophic forgetting of the original domain (French,
1999). We wish to adapt to the gender-balanced
domain without losing general translation performance. This is a particular problem when finetuning on the very small and distinct handcrafted
adaptation sets.


**2.3.1** **Regularized training**


Regularized training is a well-established approach
for minimizing catastrophic forgetting during domain adaptation of machine translation (Barone
et al., 2017). One effective form is Elastic Weight
Consolidation (EWC) (Kirkpatrick et al., 2017)
which in NMT has been shown to maintain or even

improve original domain performance (Thompson
et al., 2019; Saunders et al., 2019). In EWC a


2 The stopword list and swapping script are provided by the
authors of Zhao et al. (2018) at [https://github.com/](https://github.com/uclanlp/corefBias)
[uclanlp/corefBias](https://github.com/uclanlp/corefBias)



regularization term is added to the original log likelihood loss function _L_ when training the debiased
model (DB):


_L_ _[′]_ ( _θ_ _[DB]_ ) = _L_ ( _θ_ _[DB]_ )+ _λ_ � _F_ _j_ ( _θ_ _j_ _[DB]_ _−_ _θ_ _j_ _[B]_ [)] [2] [ (1)]

_j_


_θ_ _j_ _[B]_ [are the converged parameters of the original]
biased model, and _θ_ _[DB]_ are the current debiased
_j_
model parameters. _F_ _j_ = E� _∇_ [2] _L_ ( _θ_ _j_ _[B]_ [)] �, a Fisher
information estimate over samples from the biased
data under the biased model. We apply EWC when
performance on the original validation set drops, selecting hyperparameter _λ_ via validation set BLEU.


**2.3.2** **Gender-inflected search spaces for**
**rescoring with debiased models**


(a) A subset of flower transducer _T_ . _T_ maps vocabulary to
itself as well as to differently-gendered inflections.


(b) Acceptor _Y_ _B_ representing the biased first-pass translation
**y** **B** for source fragment ’the doctor’. The German hypothesis
has the male form.


(c) Gender-inflected search space constructed from the biased
hypothesis ‘der Arzt’. Projection of the composition _Y_ _B_ _◦_ _T_
contains paths with differently-gendered inflections of the
original biased hypothesis. This lattice can now be rescored
by a debiased model.


Figure 2: Finite State Transducers for lattice rescoring.


An alternative approach for avoiding catastrophic forgetting takes inspiration from lattice
rescoring for NMT (Stahlberg et al., 2016) and
Grammatical Error Correction (Stahlberg et al.,
2019). We assume we have two NMT models.
With one we decode fluent translations which contain gender bias ( _B_ ). For the one-best hypothesis
we would translate:


**y** **B** = argmax **y** _p_ _B_ ( **y** _|_ **x** ) (2)


The other model has undergone debiasing ( _DB_ )
at a cost to translation performance, producing:


**y** **DB** = argmax **y** _p_ _DB_ ( **y** _|_ **x** ) (3)


We construct a flower transducer _T_ that maps each
word in the target language’s vocabulary to itself,
as well as to other forms of the same word with

different gender inflections (Figure 2a). We also
construct _Y_ _B_, a lattice with one path representing
the biased but fluent hypothesis **y** **B** (Figure 2b).
The acceptor _P_ ( **y** **B** ) = proj output ( _Y_ _B_ _◦_ _T_ ) defines a language consisting of all the genderinflected versions of the biased first-pass translation
**y** **B** that are allowed by _T_ (Figure 2c). We can now
decode with lattice rescoring ( _LR_ ) by constraining
inference to _P_ ( **y** **B** ):


**y** **LR** = argmax **y** _∈P_ ( **y** **B** ) _p_ _DB_ ( **y** _|_ **x** ) (4)


In practice we use beam search to decode the various hypotheses, and construct _T_ using heuristics
on large vocabulary lists for each target language.


**3** **Experiments**


**3.1** **Languages and data**


WinoMT provides an evaluation framework for
translation from English to eight diverse languages.
We select three pairs for experiments: English to
German (en-de), English to Spanish (en-es) and
English to Hebrew (en-he). Our selection covers
three language groups with varying linguistic properties: Germanic, Romance and Semitic. Training
data available for each language pair also varies in
quantity and quality. We filter training data based
on parallel sentence lengths and length ratios.
For **en-de**, we use 17.6M sentence pairs from
WMT19 news task datasets (Barrault et al., 2019).

We validate on newstest17 and test on newstest18.

For **en-es** we use 10M sentence pairs from the
United Nations Parallel Corpus (Ziemski et al.,
2016). While still a large set, the UNCorpus exhibits far less diversity than the en-de training data.
We validate on newstest12 and test on newstest13.

For **en-he** we use 185K sentence pairs from the
multilingual TED talks corpus (Cettolo et al., 2014).
This is both a specialized domain and a much
smaller training set. We validate on the IWSLT
2012 test set and test on IWSLT 2014.

Table 1 summarises the sizes of datasets used,
including their proportion of gendered sentences
and ratio of sentences in the English source data
containing male and female stopwords. A gendered
sentence contains at least one English gendered
stopword as used by Zhao et al. (2018).
Interestingly all three datasets have about the
same proportion of gendered sentences: 11-12% of



the overall set. While en-es appears to have a much
more balanced gender ratio than the other pairs,
examining the data shows this stems largely from
sections of the UNCorpus containing phrases like
‘empower women’ and ‘violence against women’,
rather than gender-balanced professional entities.

|Col1|Training|Gendered training|M:F|Test|
|---|---|---|---|---|
|en-de<br>en-es<br>en-he|17.5M<br>10M<br>185K|2.1M<br>1.1M<br>21.4K|2.4<br>1.1<br>1.8|3K<br>3K<br>1K|



Table 1: Parallel sentence counts. A gendered sentence
pair has minimum one gendered stopword on the English side. M:F is ratio of male vs female gendered
training sentences.


For en-de and en-es we learn joint 32K BPE
vocabularies on the training data (Sennrich et al.,
2016). For en-he we use separate source and target vocabularies. The Hebrew vocabulary is a 2kmerge BPE vocabulary, following the recommendations of Ding et al. (2019) for smaller vocabularies when translating into lower-resource languages.
For the en-he source vocabulary we experimented
both with learning a new 32K vocabulary and with
reusing the joint BPE vocabulary trained on the
largest set en-de which lets us initialize the en-he
system with the pre-trained en-de model. The latter
resulted in higher BLEU and faster training.


**3.2** **Training and inference**


For all models we use a Transformer model

(Vaswani et al., 2017) with the ‘base’ parameter
settings given in Tensor2Tensor (Vaswani et al.,
2018). We train baselines to validation set BLEU
convergence on one GPU, delaying gradient updates by factor 4 to simulate 4 GPUs (Saunders
et al., 2018). During fine-tuning training is continued without learning rate resetting. Normal and
lattice-constrained decoding is via SGNMT [3] with
beam size 4. BLEU scores are calculated for cased,
detokenized output using SacreBLEU (Post, 2018)


**3.3** **Lattice rescoring with debiased models**


For lattice rescoring we require a transducer _T_ containing gender-inflected forms of words in the target vocabulary. To obtain the vocabulary for German we use all unique words in the full target training dataset. For Spanish and Hebrew, which have
smaller and less diverse training sets, we use 2018


3 [https://github.com/ucam-smt/sgnmt](https://github.com/ucam-smt/sgnmt)


OpenSubtitles word lists [4] . We then use DEMorphy
(Altinok, 2018) for German, spaCy (Honnibal and
Montani, 2017) for Spanish and the small set of
gendered suffixes for Hebrew (Schwarzwald, 1982)
to approximately lemmatize each vocabulary word
and generate its alternately-gendered forms. While
there are almost certainly paths in _T_ containing
non-words, we expect these to have low likelihood
under the debiasing models. For lattice compositions we use the efficient OpenFST implementations (Allauzen et al., 2007).


**3.4** **Results**


**3.4.1** **Baseline analysis**


In Table 2 we compare our three baselines to
commercial systems on WinoMT, using results
quoted directly from Stanovsky et al. (2019). Our
baselines achieve comparable accuracy, masculine/feminine bias score ∆ _G_ and pro/anti stereotypical bias score ∆ _S_ to four commercial translation

systems, outscoring at least one system for each
metric on each language pair.
The ∆ _S_ for our en-es baseline is surprisingly
small. Investigation shows this model predicts male
and female entities in a ratio of over 6:1. Since al
most all entities are translated as male, pro- and
anti-stereotypical class accuracy are both about
50%, making ∆ _S_ very small. This highlights the
importance of considering ∆ _S_ in the context of
∆ _G_ and M:F prediction ratio.


**3.4.2** **Counterfactual adaptation**


Table 3 compares our baseline model with the results of unregularised fine-tuning on the counterfactual sets described in Section 2.2.2.

Fine-tuning for one epoch on _original_, a subset
of the original data with gendered English stopwords, gives slight improvement in WinoMT accuracy and ∆ _G_ for all language pairs, while ∆ _S_
worsens. We suggest this set consolidates examples present in the full dataset, improving performance on gendered entities generally but emphasizing stereotypical roles.
On the _FTrans original_ set ∆ _G_ increases sharply
relative to the _original_ set, while ∆ _S_ decreases.
We suspect this set suffers from bias amplification
(Zhao et al., 2017) introduced by the baseline system during forward-translation. The model therefore over-predicts male entities even more heavily


4 Accessed Oct 2019 from [https://github.com/](https://github.com/hermitdave/FrequencyWords/)
[hermitdave/FrequencyWords/](https://github.com/hermitdave/FrequencyWords/)



than we would expect given the gender makeup of
the adaptation data’s source side. Over-predicting
male entities lowers ∆ _S_ artificially.
Adapting to _FTrans swapped_ increases accuracy
and decreases both ∆ _G_ and ∆ _S_ relative to the

baseline for en-de and en-es. This is the desired

result, but not a particularly strong one, and it is
not replicated for en-he. The _balanced_ set has a
very similar effect to the _FTrans swapped_ set, with
a smaller test BLEU difference from the baseline.

We do find that the largest improvement in
WinoMT accuracy consistently corresponds to the
model predicting male and female entities in the
closest ratio (see Appendix A). However, the best
ratios for models adapted to these datasets are 2:1
or higher, and the accuracy improvement is small.
The purpose of EWC regularization is to avoid
catastrophic forgetting of general translation ability. This does not occur in the counterfactual experiments, so we do not apply EWC. Moreover,
WinoMT accuracy gains are small with standard
fine-tuning, which allows maximum adaptation: we
suspect EWC would prevent any improvements.
Overall, improvements from fine-tuning on counterfactual datasets ( _FTrans swapped_ and _balanced_ )
are present. However, they are not very different from the improvements when fine-tuning on
equivalent non-counterfactual sets ( _original_ and
_FTrans original_ ). Improvements are also inconsistent across language pairs.


**3.4.3** **Handcrafted profession set adaptation**

Results for fine-tuning on the handcrafted set are
given in lines 3-6 of Table 4. These experiments
take place in minutes on a single GPU, compared
to several hours when fine-tuning on the counterfactual sets and far longer if training from scratch.
Fine-tuning on the handcrafted sets gives a much
faster BLEU drop than fine-tuning on counterfactual sets. This is unsurprising since the handcrafted
sets are domains of new sentences with consistent

sentence length and structure. By contrast the counterfactual sets are less repetitive and close to subsets of the original training data, slowing forgetting.
We believe the degradation here is limited only by
the ease of fitting the small handcrafted sets.
Line 4 of Table 4 adapts to the handcrafted set,
stopping when validation BLEU degrades by 5%
on each language pair. This gives a WinoMT accuracy up to 19 points above the baseline, far more
improvement than the best counterfactual result.
Difference in gender score ∆ _G_ improves by at least


|Col1|en-de<br>Acc ∆G ∆S|en-es<br>Acc ∆G ∆S|en-he<br>Acc ∆G ∆S|
|---|---|---|---|
|Microsoft<br>Google<br>Amazon<br>SYSTRAN|**74.1**<br>**0.0**<br>30.2<br>59.4<br>12.5<br>12.5<br>62.4<br>12.9<br>16.7<br>48.6<br>34.5<br>**10.3**|47.3<br>36.8<br>23.2<br>53.1<br>23.4<br>21.3<br>**59.4**<br>**15.4**<br>22.3<br>45.6<br>46.3<br>15.0|48.1<br>14.9<br>32.9<br>**53.7**<br>**7.9**<br>37.8<br>50.5<br>10.3<br>47.3<br>46.6<br>20.5<br>**24.5**|
|Baseline|60.1<br>18.6<br>13.4|49.6<br>36.7<br>**2.0**|51.3<br>15.1<br>26.4|


Table 2: WinoMT accuracy, masculine/feminine bias score ∆ _G_ and pro/anti stereotypical bias score ∆ _S_ for our
baselines compared to commercial systems, whose scores are quoted directly from Stanovsky et al. (2019).

|Col1|en-de<br>BLEU Acc ∆G ∆S|en-es<br>BLEU Acc ∆G ∆S|en-he<br>BLEU Acc ∆G ∆S|
|---|---|---|---|
|Baseline|42.7<br>60.1<br>18.6<br>13.4|27.8<br>49.6<br>36.7<br>2.0|**23.8**<br>51.3<br>15.1<br>26.4|
|Original<br>FTrans original<br>FTrans swapped<br>Balanced|41.8<br>60.7<br>15.9<br>15.6<br>43.3<br>60.0<br>20.0<br>13.9<br>**43.4**<br>63.0<br>15.4<br>12.7<br>42.5<br>**64.0**<br>**12.6**<br>**12.4**|**28.3**<br>53.0<br>**24.3**<br>10.8<br>27.4<br>51.6<br>31.6<br>-4.8<br>27.4<br>**53.7**<br>24.5<br>-3.8<br>27.7<br>52.8<br>26.2<br>**1.9**|23.5<br>**53.6**<br>**12.2**<br>31.7<br>23.4<br>48.7<br>23.0<br>**20.9**<br>23.7<br>48.1<br>20.7<br>22.7<br>**23.8**<br>48.3<br>20.8<br>24.0|



Table 3: General test set BLEU and WinoMT scores after unregularised fine-tuning the baseline on four genderbased adaptation datasets. Improvements are inconsistent across language pairs.

|Col1|en-de<br>BLEU Acc ∆G ∆S|en-es<br>BLEU Acc ∆G ∆S|en-he<br>BLEU Acc ∆G ∆S|
|---|---|---|---|
|Baseline<br>Balanced|**42.7**<br>60.1<br>18.6<br>13.4<br>42.5<br>64.0<br>12.6<br>12.4|**27.8**<br>49.6<br>36.7<br>2.0<br>27.7<br>52.8<br>26.2<br>**1.9**|23.8<br>51.3<br>15.1<br>26.4<br>23.8<br>48.3<br>20.8<br>24.0|
|~~Handcrafted~~<br>~~(no~~<br>overlap)<br>Handcrafted|40.6<br>71.2<br>3.9<br>10.6<br>40.8<br>78.3<br>**-0.7**<br>6.5|26.5<br>64.1<br>9.5<br>-10.3<br>26.7<br>68.6<br>5.2<br>-8.7|23.1<br>56.5<br>-6.2<br>28.9<br>22.9<br>65.7<br>-3.3<br>20.2|
|~~Handcrafted (con-~~<br>verged)<br>Handcrafted EWC|36.5<br>**85.3**<br>-3.2<br>6.3<br>42.2<br>74.2<br>2.2<br>8.4|25.3<br>**72.4**<br>**0.8**<br>-3.9<br>27.2<br>67.8<br>5.8<br>-8.2|22.5<br>**72.6**<br>-4.2<br>21.0<br>23.3<br>65.2<br>**-0.4**<br>25.3|
|Rescore 1 with 3<br>Rescore 1 with 4<br>Rescore 1 with 5|**42.7**<br>68.3<br>7.6<br>11.8<br>**42.7**<br>74.5<br>2.1<br>6.5<br>42.5<br>81.7<br>-2.4<br>**1.5**|**27.8**<br>62.4<br>11.1<br>-9.7<br>**27.8**<br>64.2<br>9.7<br>-10.8<br>27.7<br>68.4<br>5.6<br>-8.0|**23.9**<br>56.2<br>2.8<br>23.0<br>**23.9**<br>58.4<br>2.7<br>18.6<br>23.6<br>63.8<br>0.7<br>**12.9**|



Table 4: General test set BLEU and WinoMT scores after fine-tuning on the handcrafted profession set, compared
to fine-tuning on the most consistent counterfactual set. Lines 1-2 duplicated from Table 3. Lines 3-4 vary adaptation data. Lines 5-6 vary adaptation training procedure. Lines 7-9 apply lattice rescoring to baseline hypotheses.



a factor of 4. Stereotyping score ∆ _S_ also improves
far more than for counterfactual fine-tuning. Unlike
the Table 3 results, the improvement is consistent
across all WinoMT metrics and all language pairs.
The model adapted to no-overlap handcrafted
data (line 3) gives a similar drop in BLEU to the
model in line 4. This model also gives stronger and
more consistent WinoMT improvements over the
baseline compared to the balanced counterfactual
set, despite the implausibly strict scenario of no
English profession vocabulary in common with the
challenge set. This demonstrates that the adapted
model does not simply memorise vocabulary.
The drop in BLEU and improvement on
WinoMT can be explored by varying the training
procedure. The model of line 5 simply adapts to
handcrafted data for more iterations with no regularisation, to approximate loss convergence on
the handcrafted set. This leads to a severe drop in
BLEU, but even higher WinoMT scores.



In line 6 we regularise adaptation with EWC.
There is a trade-off between general translation
performance and WinoMT accuracy. With EWC
regularization tuned to balance validation BLEU
and WinoMT accuracy, the decrease is limited to
about 0.5 BLEU on each language pair. Adapting
to convergence, as in line 5, would lead to further
WinoMT gains at the expense of BLEU.


**3.4.4** **Lattice rescoring with debiased models**


In lines 7-9 of Table 4 we consider lattice-rescoring
the baseline output, using three models debiased
on the handcrafted data.

Line 7 rescores the general test set hypotheses
(line 1) with a model adapted to handcrafted data
that has no source language profession vocabulary
overlap with the test set (line 3). This scheme
shows no BLEU degradation from the baseline
on any language and in fact a slight improvement
on en-he. Accuracy improvements on WinoMT


|en-de<br>Acc ∆G ∆S|en-es<br>Acc ∆G ∆S|en-he<br>Acc ∆G ∆S|
|---|---|---|
|**82.0** (74.1)<br>-3.0 (0.0)<br>4.0 (30.2)<br>80.0 (59.4)<br>-3.0 (12.5)<br>**2.7** (12.5)<br>81.8 (62.4)<br>**-2.6** (12.9)<br>4.3 (16.7)<br>78.4 (48.6)<br>-4.0 (34.5)<br>5.3 (10.3)|65.8 (47.3)<br>3.8 (36.8)<br>**1.9** (23.2)<br>68.9 (53.1)<br>**0.6** (23.4)<br>4.6 (21.3)<br>**71.1** (59.4)<br>0.7 (15.4)<br>6.7 (22.3)<br>66.0 (45.6)<br>4.2 (46.3)<br>-2.1 (15.0)|63.9 (48.1)<br>-2.6 (14.9)<br>23.8 (32.9)<br>**64.6** (53.7)<br>-1.8 (7.9)<br>21.5 (37.8)<br>62.8 (50.5)<br>**-1.1** (10.3)<br>26.9 (47.3)<br>62.5 (46.6)<br>-2.0 (20.5)<br>**10.2** (24.5)|


Table 5: We generate gender-inflected lattices from commercial system translations, collected by Stanovsky et al.
(2019) (1: Microsoft, 2: Google, 3: Amazon, 4: SYSTRAN). We then rescore with the debiased model from line
5 of Table 4. Scores are for the rescored hypotheses, with bracketed baseline scores duplicated from Table 2.



are only slightly lower than for decoding with the
rescoring model directly, as in line 3.
In line 8, lattice rescoring with the nonconverged model adapted to handcrafted data (line
4) likewise leaves general BLEU unchanged or
slightly improved. When lattice rescoring the
WinoMT challenge set, 79%, 76% and 49% of
the accuracy improvement is maintained on en-de,
en-es and en-he respectively. This corresponds to
accuracy gains of up to 30% relative to the baselines with no general translation performance loss.
In line 9, lattice-rescoring with the converged
model of line 5 limits BLEU degradation to 0.2
BLEU on all languages, while maintaining 85%,
82% and 58% of the WinoMT accuracy improvement from the converged model for the three language pairs. Lattice rescoring with this model gives
accuracy improvements over the baseline of 36%,
38% and 24% for en-de, en-es and en-he.
Rescoring en-he maintains a much smaller proportion of WinoMT accuracy improvement than
en-de and en-es. We believe this is because the

en-he baseline is particularly weak, due to a small
and non-diverse training set. The baseline must
produce some inflection of the correct entity before
lattice rescoring can have an effect on gender bias.


**3.4.5** **Reducing gender bias in ‘black box’**
**commercial systems**

Finally, in Table 5, we apply the gender inflection
transducer to the commercial system translations [5]

listed in Table 2. We find rescoring these lattices
with our strongest debiasing model (line 5 of Table
4) substantially improves WinoMT accuracy for all
systems and language pairs.
One interesting observation is that WinoMT accuracy after rescoring tends to fall in a fairly narrow range for each language relative to the performance range of the baseline systems. For example, a 25.5% range in baseline en-de accuracy


5 The raw commercial system translations are provided by
the authors of Stanovsky et al. (2019) at [https://github.](https://github.com/gabrielStanovsky/mt_gender)
[com/gabrielStanovsky/mt_gender](https://github.com/gabrielStanovsky/mt_gender)



becomes a 3.6% range after rescoring. This suggests that our rescoring approach is not limited as
much by the bias level of the baseline system as
by the gender-inflection transducer and the models used in rescoring. Indeed, we emphasise that
the large improvements reported in Table 5 do not
require any knowledge of the commercial systems
or the data they were trained on; we use only the
translation hypotheses they produce and our own
rescoring model and transducer.


**4** **Conclusions**


We treat the presence of gender bias in NMT systems as a domain adaptation problem. We demonstrate strong improvements under the WinoMT
challenge set by adapting to tiny, handcrafted
gender-balanced datasets for three language pairs.
While naive domain adaptation leads to catastrophic forgetting, we further demonstrate two approaches to limit this: EWC and a lattice rescoring
approach. Both allow debiasing while maintaining
general translation performance. Lattice rescoring,
although a two-step procedure, allows far more
debiasing and potentially no degradation, without
requiring access to the original model.
We suggest small-domain adaptation as a more
effective and efficient approach to debiasing machine translation than counterfactual data augmentation. We do not claim to fix the bias problem
in NMT, but demonstrate that bias can be reduced
without degradation in overall translation quality.


**Acknowledgments**


This work was supported by EPSRC grants
EP/M508007/1 and EP/N509620/1 and has been

performed using resources provided by the Cambridge Tier-2 system operated by the University of
Cambridge Research Computing Service [6] funded
by EPSRC Tier-2 capital grant EP/P020259/1.


6 [http://www.hpc.cam.ac.uk](http://www.hpc.cam.ac.uk)


**References**


Lauren Ackerman. 2019. Syntactic and cognitive issues in investigating gendered coreference. _Glossa:_
_a journal of general linguistics_, 4(1).


Cyril Allauzen, Michael Riley, Johan Schalkwyk, Wojciech Skut, and Mehryar Mohri. 2007. OpenFst: A
general and efficient weighted finite-state transducer
library. In _International Conference on Implemen-_
_tation and Application of Automata_, pages 11–23.
Springer.


Duygu Altinok. 2018. DEMorphy, German language morphological analyzer. _arXiv preprint_
_arXiv:1803.00902_ .


Antonio Valerio Miceli Barone, Barry Haddow, Ulrich
[Germann, and Rico Sennrich. 2017. Regularization](https://doi.org/10.18653/v1/D17-1156)
[techniques for fine-tuning in neural machine trans-](https://doi.org/10.18653/v1/D17-1156)
[lation. In](https://doi.org/10.18653/v1/D17-1156) _Proceedings of the 2017 Conference on_
_Empirical Methods in Natural Language Processing_,
pages 1489–1494, Copenhagen, Denmark. Association for Computational Linguistics.


Lo¨ıc Barrault, Ondˇrej Bojar, Marta R. Costa-juss`a,
Christian Federmann, Mark Fishel, Yvette Graham, Barry Haddow, Matthias Huck, Philipp Koehn,
Shervin Malmasi, Christof Monz, Mathias M¨uller,
Santanu Pal, Matt Post, and Marcos Zampieri. 2019.
[Findings of the 2019 conference on machine transla-](https://doi.org/10.18653/v1/W19-5301)
[tion (WMT19). In](https://doi.org/10.18653/v1/W19-5301) _Proceedings of the Fourth Con-_
_ference on Machine Translation (Volume 2: Shared_
_Task Papers, Day 1)_, pages 1–61, Florence, Italy. Association for Computational Linguistics.


Tolga Bolukbasi, Kai-Wei Chang, James Y Zou,
Venkatesh Saligrama, and Adam T Kalai. 2016.
Man is to computer programmer as woman is to
homemaker? Debiasing word embeddings. In _Ad-_
_vances in neural information processing systems_,
pages 4349–4357.


Mauro Cettolo, Jan Niehues, Sebastian St¨uker, Luisa
Bentivogli, and Marcello Federico. 2014. Report
on the 11th IWSLT evaluation campaign, IWSLT
2014. In _Proceedings of the International Workshop_
_on Spoken Language Translation, Hanoi, Vietnam_,
page 57.


Shuoyang Ding, Adithya Renduchintala, and Kevin
[Duh. 2019. A call for prudent choice of subword](https://www.aclweb.org/anthology/W19-6620)
[merge operations in neural machine translation. In](https://www.aclweb.org/anthology/W19-6620)
_Proceedings of Machine Translation Summit XVII_
_Volume 1: Research Track_, pages 204–213, Dublin,
Ireland. European Association for Machine Translation.


Joel Escud´e Font and Marta R. Costa-juss`a. 2019.

[Equalizing gender bias in neural machine transla-](https://doi.org/10.18653/v1/W19-3821)
[tion with word embeddings techniques. In](https://doi.org/10.18653/v1/W19-3821) _Proceed-_
_ings of the First Workshop on Gender Bias in Natu-_
_ral Language Processing_, pages 147–154, Florence,
Italy. Association for Computational Linguistics.



M. Amin Farajian, Marco Turchi, Matteo Negri, and
Marcello Federico. 2017. [Multi-domain neural](https://doi.org/10.18653/v1/W17-4713)
[machine translation through unsupervised adapta-](https://doi.org/10.18653/v1/W17-4713)
[tion. In](https://doi.org/10.18653/v1/W17-4713) _Proceedings of the Second Conference on_
_Machine Translation_, pages 127–137, Copenhagen,
Denmark. Association for Computational Linguistics.


Markus Freitag and Yaser Al-Onaizan. 2016. Fast
domain adaptation for Neural Machine Translation.
_CoRR_, abs/1612.06897.


Robert M French. 1999. Catastrophic forgetting in connectionist networks. _Trends in cognitive sciences_,
3(4):128–135.


Hila Gonen and Yoav Goldberg. 2019. Lipstick on a
pig: Debiasing methods cover up systematic gender
biases in word embeddings but do not remove them.
In _Proceedings of the 2019 Conference of the North_
_American Chapter of the Association for Computa-_
_tional Linguistics: Human Language Technologies,_
_Volume 1 (Long and Short Papers)_, pages 609–614.


AI HLEG. 2019. _Ethics guidelines for trustworthy AI_ .
High-Level Expert Group on Artificial Intelligence.


Matthew Honnibal and Ines Montani. 2017. spaCy 2:
Natural language understanding with bloom embeddings. _Convolutional Neural Networks and Incre-_
_mental Parsing_ .


[Dirk Hovy and Shannon L. Spruit. 2016. The social](https://doi.org/10.18653/v1/P16-2096)
[impact of natural language processing. In](https://doi.org/10.18653/v1/P16-2096) _Proceed-_
_ings of the 54th Annual Meeting of the Association_
_for Computational Linguistics (Volume 2: Short Pa-_
_pers)_, pages 591–598, Berlin, Germany. Association
for Computational Linguistics.


James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz,
Joel Veness, Guillaume Desjardins, Andrei A Rusu,
Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. 2017. Overcoming catastrophic forgetting in neural networks. _Pro-_
_ceedings of the National Academy of Sciences of the_
_United States of America_, 114(13):3521–3526.


Kaiji Lu, Piotr Mardziel, Fangjing Wu, Preetam Amancharla, and Anupam Datta. 2018. Gender bias in
neural natural language processing. _arXiv preprint_
_arXiv:1807.11714_ .


[Paul Michel and Graham Neubig. 2018. Extreme adap-](https://doi.org/10.18653/v1/P18-2050)
[tation for personalized neural machine translation.](https://doi.org/10.18653/v1/P18-2050)
In _Proceedings of the 56th Annual Meeting of the_
_Association for Computational Linguistics (Volume_
_2: Short Papers)_, pages 312–318, Melbourne, Australia. Association for Computational Linguistics.


Julia Misersky, Asifa Majid, and Tineke M Snijders.
2019. Grammatical gender in German influences
how role-nouns are interpreted: Evidence from erps.
_Discourse Processes_, 56(8):643–654.


Amit Moryossef, Roee Aharoni, and Yoav Goldberg.
[2019. Filling gender & number gaps in neural ma-](https://doi.org/10.18653/v1/W19-3807)
[chine translation with black-box context injection.](https://doi.org/10.18653/v1/W19-3807)
In _Proceedings of the First Workshop on Gender_
_Bias in Natural Language Processing_, pages 49–54,
Florence, Italy. Association for Computational Linguistics.


Kishore Papineni, Salim Roukos, Todd Ward, and Wei[Jing Zhu. 2002. BLEU: a method for automatic eval-](https://doi.org/10.3115/1073083.1073135)
[uation of machine translation.](https://doi.org/10.3115/1073083.1073135) In _Proceedings of_
_the 40th Annual Meeting of the Association for Com-_
_putational Linguistics_, pages 311–318, Philadelphia,
Pennsylvania, USA. Association for Computational
Linguistics.


[Ji Ho Park, Jamin Shin, and Pascale Fung. 2018. Re-](https://doi.org/10.18653/v1/D18-1302)
[ducing gender bias in abusive language detection.](https://doi.org/10.18653/v1/D18-1302)
In _Proceedings of the 2018 Conference on Em-_
_pirical Methods in Natural Language Processing_,
pages 2799–2804, Brussels, Belgium. Association
for Computational Linguistics.


[Matt Post. 2018. A call for clarity in reporting BLEU](https://doi.org/10.18653/v1/W18-6319)
[scores. In](https://doi.org/10.18653/v1/W18-6319) _Proceedings of the Third Conference on_
_Machine Translation: Research Papers_, pages 186–
191, Belgium, Brussels. Association for Computational Linguistics.


Marcelo OR Prates, Pedro H Avelar, and Lu´ıs C Lamb.
2019. Assessing gender bias in machine translation:
a case study with google translate. _Neural Comput-_
_ing and Applications_, pages 1–19.


Rachel Rudinger, Jason Naradowsky, Brian Leonard,
and Benjamin Van Durme. 2018. [Gender bias in](https://doi.org/10.18653/v1/N18-2002)
[coreference resolution. In](https://doi.org/10.18653/v1/N18-2002) _Proceedings of the 2018_
_Conference of the North American Chapter of the_
_Association for Computational Linguistics: Human_
_Language Technologies, Volume 2 (Short Papers)_,
pages 8–14, New Orleans, Louisiana. Association
for Computational Linguistics.


Danielle Saunders, Felix Stahlberg, Adri`a de Gispert,
and Bill Byrne. 2018. [Multi-representation en-](https://doi.org/10.18653/v1/P18-2051)
[sembles and delayed SGD updates improve syntax-](https://doi.org/10.18653/v1/P18-2051)
[based NMT.](https://doi.org/10.18653/v1/P18-2051) In _Proceedings of the 56th Annual_
_Meeting of the Association for Computational Lin-_
_guistics (Volume 2:_ _Short Papers)_, pages 319–
325, Melbourne, Australia. Association for Computational Linguistics.


Danielle Saunders, Felix Stahlberg, Adri`a de Gispert,
[and Bill Byrne. 2019. Domain adaptive inference](https://doi.org/10.18653/v1/P19-1022)
[for neural machine translation. In](https://doi.org/10.18653/v1/P19-1022) _Proceedings of_
_the 57th Annual Meeting of the Association for Com-_
_putational Linguistics_, pages 222–228, Florence,
Italy. Association for Computational Linguistics.


Ora Schwarzwald. 1982. Feminine formation in modern Hebrew. _Hebrew Annual Review_, 6:153–178.


Rico Sennrich, Barry Haddow, and Alexandra Birch.
2016. [Neural machine translation of rare words](https://doi.org/10.18653/v1/P16-1162)
[with subword units. In](https://doi.org/10.18653/v1/P16-1162) _Proceedings of the 54th An-_
_nual Meeting of the Association for Computational_



_Linguistics (Volume 1: Long Papers)_, pages 1715–
1725, Berlin, Germany. Association for Computational Linguistics.


Felix Stahlberg, Christopher Bryant, and Bill Byrne.
2019. Neural grammatical error correction with finite state transducers. In _Proceedings of the 2019_
_Conference of the North American Chapter of the_
_Association for Computational Linguistics: Human_
_Language Technologies, Volume 1 (Long and Short_
_Papers)_, pages 4033–4039.


Felix Stahlberg, Eva Hasler, Aurelien Waite, and Bill
[Byrne. 2016. Syntactically guided neural machine](https://doi.org/10.18653/v1/P16-2049)
[translation. In](https://doi.org/10.18653/v1/P16-2049) _Proceedings of the 54th Annual Meet-_
_ing of the Association for Computational Linguistics_
_(Volume 2: Short Papers)_, pages 299–305, Berlin,
Germany. Association for Computational Linguistics.


Gabriel Stanovsky, Noah A. Smith, and Luke Zettlemoyer. 2019. [Evaluating gender bias in machine](https://www.aclweb.org/anthology/P19-1164)
[translation. In](https://www.aclweb.org/anthology/P19-1164) _Proceedings of the 57th Annual Meet-_
_ing of the Association for Computational Linguistics_,
pages 1679–1684, Florence, Italy. Association for
Computational Linguistics.


Tony Sun, Andrew Gaut, Shirlyn Tang, Yuxin Huang,
Mai ElSherief, Jieyu Zhao, Diba Mirza, Elizabeth
Belding, Kai-Wei Chang, and William Yang Wang.
2019. [Mitigating gender bias in natural language](https://doi.org/10.18653/v1/P19-1159)
[processing: Literature review.](https://doi.org/10.18653/v1/P19-1159) In _Proceedings of_
_the 57th Annual Meeting of the Association for Com-_
_putational Linguistics_, pages 1630–1640, Florence,
Italy. Association for Computational Linguistics.


Brian Thompson, Jeremy Gwinnup, Huda Khayrallah,
[Kevin Duh, and Philipp Koehn. 2019. Overcoming](https://doi.org/10.18653/v1/N19-1209)
[catastrophic forgetting during domain adaptation of](https://doi.org/10.18653/v1/N19-1209)
[neural machine translation. In](https://doi.org/10.18653/v1/N19-1209) _Proceedings of the_
_2019 Conference of the North American Chapter of_
_the Association for Computational Linguistics: Hu-_
_man Language Technologies, Volume 1 (Long and_
_Short Papers)_, pages 2062–2068, Minneapolis, Minnesota. Association for Computational Linguistics.


Eva Vanmassenhove, Christian Hardmeier, and Andy
[Way. 2018. Getting gender right in neural machine](https://doi.org/10.18653/v1/D18-1334)
[translation. In](https://doi.org/10.18653/v1/D18-1334) _Proceedings of the 2018 Conference_
_on Empirical Methods in Natural Language Process-_
_ing_, pages 3003–3008, Brussels, Belgium. Association for Computational Linguistics.


Ashish Vaswani, Samy Bengio, Eugene Brevdo, Francois Chollet, Aidan Gomez, Stephan Gouws, Llion
Jones, Łukasz Kaiser, Nal Kalchbrenner, Niki Parmar, Ryan Sepassi, Noam Shazeer, and Jakob Uszko[reit. 2018. Tensor2Tensor for neural machine trans-](https://www.aclweb.org/anthology/W18-1819)
[lation. In](https://www.aclweb.org/anthology/W18-1819) _Proceedings of the 13th Conference of the_
_Association for Machine Translation in the Ameri-_
_cas (Volume 1: Research Papers)_, pages 193–199,
Boston, MA. Association for Machine Translation
in the Americas.


Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob
Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz


Kaiser, and Illia Polosukhin. 2017. Attention is all
you need. In _Advances in Neural Information Pro-_
_cessing Systems_, pages 6000–6010.


Wei Wang, Taro Watanabe, Macduff Hughes, Tetsuji
Nakagawa, and Ciprian Chelba. 2018. [Denois-](https://doi.org/10.18653/v1/W18-6314)
[ing neural machine translation training with trusted](https://doi.org/10.18653/v1/W18-6314)
[data and online data selection.](https://doi.org/10.18653/v1/W18-6314) In _Proceedings of_
_the Third Conference on Machine Translation: Re-_
_search Papers_, pages 133–143, Belgium, Brussels.
Association for Computational Linguistics.


Jieyu Zhao, Tianlu Wang, Mark Yatskar, Vicente Ordonez, and Kai-Wei Chang. 2017. [Men also like](https://doi.org/10.18653/v1/D17-1323)
[shopping: Reducing gender bias amplification using](https://doi.org/10.18653/v1/D17-1323)
[corpus-level constraints. In](https://doi.org/10.18653/v1/D17-1323) _Proceedings of the 2017_
_Conference on Empirical Methods in Natural Lan-_
_guage Processing_, pages 2979–2989, Copenhagen,
Denmark. Association for Computational Linguistics.


Jieyu Zhao, Tianlu Wang, Mark Yatskar, Vicente Or[donez, and Kai-Wei Chang. 2018. Gender bias in](https://doi.org/10.18653/v1/N18-2003)
coreference resolution: [Evaluation and debiasing](https://doi.org/10.18653/v1/N18-2003)
[methods.](https://doi.org/10.18653/v1/N18-2003) In _Proceedings of the 2018 Conference_
_of the North American Chapter of the Association_
_for Computational Linguistics: Human Language_
_Technologies, Volume 2 (Short Papers)_, pages 15–20,
New Orleans, Louisiana. Association for Computational Linguistics.


Michał Ziemski, Marcin Junczys-Dowmunt, and Bruno
Pouliquen. 2016. [The united nations parallel cor-](https://www.aclweb.org/anthology/L16-1561)
[pus v1.0.](https://www.aclweb.org/anthology/L16-1561) In _Proceedings of the Tenth Interna-_
_tional Conference on Language Resources and Eval-_
_uation (LREC 2016)_, pages 3530–3534, Portoroˇz,
Slovenia. European Language Resources Association (ELRA).


Ran Zmigrod, Sabrina J. Mielke, Hanna Wallach, and
[Ryan Cotterell. 2019. Counterfactual data augmen-](https://doi.org/10.18653/v1/P19-1161)
[tation for mitigating gender stereotypes in languages](https://doi.org/10.18653/v1/P19-1161)
[with rich morphology. In](https://doi.org/10.18653/v1/P19-1161) _Proceedings of the 57th_
_Annual Meeting of the Association for Computa-_
_tional Linguistics_, pages 1651–1661, Florence, Italy.
Association for Computational Linguistics.


**A** **WinoMT male:female prediction ratio**


We report ∆ _G_ on WinoMT for easy comparison
to previous work, but also find that M:F prediction
ratio on WinoMT is an intuitive and interesting
metric. Tables 6 and 7 expand on the results of
Tables 3 and 4 respectively.


|Col1|en-de<br>BLEU Acc M:F|en-es<br>BLEU Acc M:F|en-he<br>BLEU Acc M:F|
|---|---|---|---|
|Baseline|42.7<br>60.1<br>3.4|27.8<br>49.6<br>6.3|**23.8**<br>51.3<br>2.2|
|Original<br>FTrans original<br>FTrans swapped<br>Balanced|41.8<br>60.7<br>3.1<br>43.3<br>60.0<br>3.9<br>**43.4**<br>63.0<br>3.1<br>42.5<br>**64.0**<br>**2.7**|**28.3**<br>53.0<br>**4.0**<br>27.4<br>51.6<br>5.4<br>27.4<br>**53.7**<br>**4.0**<br>27.7<br>52.8<br>4.3|23.5<br>**53.6**<br>**2.0**<br>23.4<br>48.7<br>3.0<br>23.7<br>48.1<br>2.6<br>**23.8**<br>48.3<br>2.7|


Table 6: General test set BLEU and WinoMT scores after unregularised fine-tuning the baseline on four genderbased adaptation datasets.

|Col1|en-de<br>BLEU Acc M:F|en-es<br>BLEU Acc M:F|en-he<br>BLEU Acc M:F|
|---|---|---|---|
|Baseline<br>Balanced|**42.7**<br>60.1<br>3.4<br>42.5<br>64.0<br>2.7|**27.8**<br>49.6<br>6.3<br>27.7<br>52.8<br>4.3|23.8<br>51.3<br>2.2<br>23.8<br>48.3<br>2.7|
|Handcrafted (no overlap)<br>Handcrafted|40.6<br>71.2<br>1.7<br>40.8<br>78.3<br>1.3|26.5<br>64.1<br>2.4<br>26.7<br>68.6<br>1.9|23.1<br>56.5<br>0.8<br>22.9<br>65.7<br>0.9|
|Handcrafted (converged)<br>Handcrafted EWC|36.5<br>**85.3**<br>**0.9**<br>42.2<br>74.2<br>1.6|25.3<br>**72.4**<br>**1.5**<br>27.2<br>67.8<br>2.0|22.5<br>**72.6**<br>**1.0**<br>23.3<br>65.2<br>1.2|
|Rescore 1 with 3<br>Rescore 1 with 4<br>Rescore 1 with 5|**42.7**<br>68.3<br>2.2<br>**42.7**<br>74.5<br>1.6<br>42.5<br>81.7<br>**1.1**|**27.8**<br>62.4<br>2.3<br>**27.8**<br>64.2<br>2.1<br>27.7<br>68.4<br>1.8|**23.9**<br>56.2<br>1.3<br>**23.9**<br>58.4<br>1.3<br>23.6<br>63.8<br>1.3|



Table 7: General test set BLEU and WinoMT scores after fine-tuning on the handcrafted profession set, compared
to fine-tuning on the most consistent counterfactual set. Lines 1-2 duplicated from Table 6. Lines 3-4 vary adaptation data. Lines 5-6 vary adaptation training procedure. Lines 7-9 apply lattice rescoring to baseline hypotheses.


