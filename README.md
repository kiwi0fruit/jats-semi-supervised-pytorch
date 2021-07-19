Applying VAE and DGM families to JATS database in PyTorch
=========================================================

To run:

* [Install work environment via conda](https://github.com/kiwi0fruit/pyappshare/tree/master/template_env)
* [Setup mypy and pylint linters in Visual Studio Code and PyCharm](./README_SETUP_LINTERS.md)
* Open this folder in Visual Studio Code, open and run "main.py" in Data Science mode via
  [Python extension for Visual Studio Code](https://marketplace.visualstudio.com/items?itemName=ms-python.python).

This would have worked if I published SOLTI database together with this repo. But currently it's not possible.
But if you are interested researcher you can contact me here on GitHub and [Andrew Khizhnyak](https://vk.com/hizhnjak) for permission.

I forked some repos, made them work together, fixed bugs when necessary and added types annotations:

* [beta-tcvae](https://github.com/rtqichen/beta-tcvae)
* [normalizing-flows](https://github.com/tonyduan/normalizing-flows) and [BNAF](https://github.com/nicola-decao/BNAF)
* [semi-supervised-pytorch](https://github.com/wohlert/semi-supervised-pytorch)


First models (previous version)
----------------------------------------------

*Repo with code for this version is available [here](https://github.com/kiwi0fruit/jats-semi-supervised-pytorch/tree/0.1).*

This repo contains code that is my playground for applying VAE family models to [JATS](https://github.com/kiwi0fruit/jats) SOLTI-160 database
by Andrew Khizhnyak based on survey by Victor Talanov.

I started with [VAE](https://arxiv.org/abs/1312.6114), then tried [MMD-AE](https://arxiv.org/abs/1706.02262),
then [Beta TC VAE](https://arxiv.org/abs/1802.04942). I also used [PCA and FA](./vae/linear_component_analyzer.py) for comparison.

Research started with hope to create a latent variables model that would be better than PCA and FA somehow.

In search for better NLL (given by BCE) of the reconstructed data than in VAE and Beta TC VAE I tried MMD-AE - as even FA gave much better NLL that VAE
(I'm aware that comparison via NLL only is not fare as with FA we don't have NELBO for proper comparison. But still...).
MMD-AE gave nice NLL comparable with FA but gave random latents. Beta TC VAE gave more stable latents but with a cost of high NLL.
First trainig with Beta TC VAE then training the same model with MMD-AE allowed to train some models then pick the most "common" one.
This way I got resemblance of latents stability and nice NLL (with the same z dim the same as in FA).
Still the results were not that different from FA results so I tried [ADGM](https://arxiv.org/abs/1602.05473).

DGM family fits perfectly as SOLTI database has "types" assigned to each profile via two ways (self providing type in the survey + result of the survey).
As both are not enough reliable in my opinion it's better to take only types than coinside. Hence we have partially labelled database.
This gave more interesting results in my opinion but I'm still in the process of investigating them.

I split the database into learn part and test part as 9:1. Database contains 6406 profiles with 162 questions each.

Worth mentioning that I decided to have a weighted random sampler that would sample some profiles much more frequently.
I decided to use types results of the test for this (together with sex). Presumably it's necessary as some types
like to take psychological surveys much more than other types.

There are 3 straghtforard ways to get labels: 16 types, 8 dominant functions, 4 JATS temperaments. I tried all three and dumped raw results to
[output-old](./output_old) folder. There are some svg stats there (see [labels colors meanings](./output_old/types_colors.svg) for details).

Worth mentioning classifiation accuracy that ADGM gives:

* 16 Types: 95% for learn data, 80% for test data,
* 8 Dominant functions: 95% for learn data, 85% for test data,
* 4 JATS temperaments: 92% for leard data, 83% for test data.
* TODO: make learn and lest coniside.

NLL is still worse than in FA but not much. And latents are quite different as the model uses labels.

At some point I also played with normalizing flows added to VAE but abandonded it.


Second unfinished model (current version)
----------------------------------------------
After the first attempt I switched completely to the Beta-TC-VAE models. Andrew Khizhnyak provided me with additional \~4000 data points total of \~10000 now. I re-split the database into learn part and test part as 7:3.

After reading papers [1 - Disentangling Disentanglement in Variational Autoencoders](https://arxiv.org/abs/1812.02833) and [2 - Variational Autoencoders Pursue PCA Directions (by Accident)](https://arxiv.org/abs/1812.06775) I 
assumed that classical VAEs do not guarantee the disentanglement (I sometimes got VAE-trained models with linearly correlated latents when NELBO had as nice values as in models without correlated latents. Hence no guarantee. This behaviour was mentioned in the 1st parer) but Beta-VAE and Beta-TC-VAE (beta total correlation VAE) either do not converge to the same values of NLL or do it too slow.

Hence I implemented hybrid Beta-TC-VAE model that consists of two Beta-TC-VAEs that share decoders and encoding of mean values. But sigma values are encoded partially separately. The first model gives NELBO1 with high beta value, the second one gives NELBO2 with low beta value. The total error is a weighted sum of NELBO1 and NELBO2. On the first etape NELBO2 has more weight but then NELBO1 has more weight. This way I got the model that always converges to NLL values not worse than in Factor Analysis. And has low total correlation value estimates that always give low latents linear correlations (as an easy control parameter). I also used MMD error on encoder means to prevent prior collapse that gives correlated latent means but indepernent samples (normal distributed samples from means and sigmas) - that happens if the dimension has high enough sigmas for most of the dataset. Unfortumnately even with this hybrid model Beta-TC-VAE methods do not allow to get more than 8-dim latents on this database.

But even this is not enough to get unique latents as with standard normal distributed priors all rotations of the latents space are equally good for optimizer (as mentioned in the 1st and the 2nd papers). Instead of trying other priors I desided to use the fact that the database is partially labelled. 16 labells can be split into binary classes so I can use them to split the databade in two parts in different ways. Hence axes can be aligned using these classes and simple sigmoid classifiers. Some alignments impede learning a model with nice NLL and uncorrelated latents. But some do not impede this in any way. With trial and error I chose not impeding "FZA4" latents (described [*here*](https://github.com/kiwi0fruit/jats#118-fza4-hypothesis-for-8-axes-of-independent-variation-in-factorized-traits-space) in accordance with the framework of existing ideas about psychological traits included in the questionnaire).

This semi-supervised axes alignment converges to two or three attractors when learning. The first one is the most common. I simply chose it as a desired stable output (aka "FZA4" latents). It still has slight variations in latents hence I used 6 independently learned models and averaged their encoders output of means (mus). After that I excluded encoder from the optimizer (except sigmas). See averaged "FZA4" latents in the [output](./output) folder (see legend description [here](https://github.com/kiwi0fruit/jats#119-brightdeep-rainbow-colors-to-plot-16-probability-density-functions-on-a-single-image)). 

"FZA4" latents have some interpretation in accordance with the framework of existing ideas about psychological traits (see above) but this interpretation is somewhat lacking. This is mostly because framework of existing interpretations uses convenient non-independant observables but the model tries to learn independent latents. Hence I added a subdecoder that maps 8 independent latents to 12 non-independent parameters that have a more convenient interpretation (see [Khizhnyak axes](https://github.com/kiwi0fruit/jats#112-khizhnyak-functions-and-axes)). I again used axes alignment using sigmoid functions for this. Surprisingly adding this subdecoder made reconstruction error significantly better. Without it 8-dim NLL from VAE was as 8-dim NLL in Factor analysis. But after adding this subdecoder 8-dim reconstruction error from VAE become as 9-dim reconstruction error in Factor analysis. Not much but significant. More surprisingly adding subdecoder without axes alignment or using other theoretically possible axes alignments do not make reconstruction error better. This means that Khizhnyak axes are special somehow.

Unfortunately 12 intermediate axes are dependent hence they are unstable to learn. At the moment I came up with the learning regime that makes learning them more stable without harm to reconstruction error. Is it still unsatisfying but I hope that averaging several models would not impede reconstruction error. But it's not guaranteed as _dependent_ axes can be "like" snake that swallowed a fresh elephant. And several independently-learned models can be like snakes with fresh elephants in different places of their gastrointestinal tract. And averaging them would not give me again a snake that swallowed an elephant (but already digested it) hence this might impede the reconstruction error. But I still hope that averaging will make do.

Mappings:

(Encoder) 161 dim [questionnaire] => 8 mu, 8 sigma => (Sampler) 8 dim [FZA4] => (SubDecoder) 12 dim [Khizhnyak axes] => 160 dim [questionnaire]
