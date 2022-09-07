---
title: 'Tagging Latency Estimator: A Standalone Software for Estimating Latency of Event-Related Potentials in P300-based Brain-Computer Interfaces'
tags:
  - Unity
  - C#
  - Event-Related Potential (ERP)
  - P300
  - Brain-Computer Interface (BCI)
authors:
  - name: Grégoire H. Cattan
    orcid: 0000-0002-7515-0690
    affiliation: 1
  - name: Cesar Mendoza
    affiliation: 2
affiliations:
 - name: IBM, Cloud and Cognitive Software, Poland
   index: 1
 - name: IHMTEK, France
   index: 2
date: 28 August 2021
bibliography: paper.bib

---

# Summary

Event-related potentials (ERPs) are small potentials elicited by the brain in response to an external stimulation. They are measured using an electroencephalogram (EEG). Differences in the onset time and amplitude of ERPs reflect different sensory and high-level brain processing functions, such as the recognition of symbols, the correctness of presented information, or changes in a subject's attention [@Luck:2012]. For these reasons, ERPs are a useful tool for describing the processing of information inside the brain, with practical applications in the domain of brain-computer interfaces [@Wolpaw:2012]. 

To detect and evaluate an ERP in an ongoing electroencephalogram (EEG), it is necessary to tag the EEG with the exact onset time of the stimulus. A precise hardware method is then used to assess the latency between the tag and the exact onset of the stimulus on screen [@Andreev:2019]. This methods relies on a photodiode, placed in front of a stimulus which record the exact moment where the stimulus actually lighten on the screen. The latency is then computed by substracting the time when the EEG is tagged to the actual apparition of the stimulus on screen. 

A fixed latency engenders a constant offset which can be easily removed. However, the failure to control the tagging pipeline causes problems when interpreting ERPs thus leading to contradictory conclusions [@Amin:2015; @Käthner:2015; @Pegna:2018] - such as confunding two ERPs. This is particulary true when comparing ERPs elicited by stimuli presented on different platforms as these platforms usually introduce latencies that differ due to specific hardware and software configurations [@Cattan:2021]. Another common problem, is the display of stimuli that don't match the position of the photodiode. In fact, different stimuli have different latencies as different parts of the screen don't refresh at the same time. 

Analysis of the tagging pipeline [@Cattan:2018] have led to the development of a theorical framework to interpret and eventually correct the measured latency, based on high-level configuration, such as the position of the photodiode (if known), the distribution of the stimuli on screen, the screen orientation or the number of cameras within the screen - like in virtual reality where the screen is split in two.

# Statement of need

`TaggingLatencyEstimator` is a standalone software developed in Unity which provides a C# implementation for @Cattan:2018. 

As briefly summarized in the [Summary subsection](#summary), the complexity of the tagging pipeline is a problem under-estimated in the scientific litterature, which could lead to the misinterpretation of the ongoing brain processing functions. To our knowledge, there is no software or tools which may facilitate the correct estimation and interpretation of such latency. 

Based on the model described in @Cattan:2018, an early version of this software was used in @korczowski:2019a; @korczowski:2019b; @korczowski:2019c; @korczowski:2019d; @Vaineau:2019; @VanVeen:2019; @Cattan:2019; @Cattan:2021 thereby outlining the need for such an implementation.

# References
