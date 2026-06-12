---
layout: page
title: Research
subtitle: Publications, history, and the ideas behind IMPSY.
permalink: /research/
description: Publications and research history of the IMPSY (Intelligent Musical Prediction System) project, from the early RoboJam MDRNN work through to the 2026 design-space paper and the web interface.
---

IMPSY sits in a research thread that began with mixture density recurrent networks for musical interaction and has grown into a general toolkit for intelligent instruments. The papers below cover both core IMPSY development and work by students and collaborators that builds on the platform.

## Core IMPSY publications

<ul class="pub-list">
  <li>
    <div class="pub-year">2026</div>
    <div class="pub-title">Opening the Design Space: Two Years of Performance with Intelligent Musical Instruments</div>
    <p class="mb-2 text-body-secondary">Charles Martin. <em>International Conference on New Interfaces for Musical Expression (NIME)</em>, London. Introduces the Raspberry-Pi-based IMPSY platform and reflects on a two-year first-person artistic research process with five prototype instruments (Intelligent Volca, MicroFreak, S-1, DAW, Setup). Argues that remapping can substitute for retraining, that fast input interleaving is a viable co-creative strategy, and that small-data AI models are a portable design resource.</p>
    <a href="https://arxiv.org/abs/2604.23583">arXiv →</a>
  </li>
  <li>
    <div class="pub-year">2022</div>
    <div class="pub-title">Performing with a Generative Electronic Music Controller</div>
    <p class="mb-2 text-body-secondary">Charles Martin. <em>Joint Proceedings of the ACM IUI Workshops</em>. A reflection on using the IMPS prediction system in live electronic music performance, prefiguring the design directions later formalised in IMPSY.</p>
    <a href="https://ceur-ws.org/Vol-3124/paper10.pdf">PDF →</a>
  </li>
  <li>
    <div class="pub-year">2020</div>
    <div class="pub-title">Understanding Musical Predictions with an Embodied Interface for Musical Machine Learning</div>
    <p class="mb-2 text-body-secondary">Charles Martin, Kyrre Glette, Tønnes Nygaard, and Jim Torresen. <em>Frontiers in Artificial Intelligence</em>. A study of embodied musical prediction using the EMPI hardware controller: how performers experience, interpret, and play with predictive output from an MDRNN.</p>
    <a href="https://doi.org/10.3389/frai.2020.00006">DOI →</a>
  </li>
  <li>
    <div class="pub-year">2019</div>
    <div class="pub-title">An Interactive Musical Prediction System with Mixture Density Recurrent Neural Networks</div>
    <p class="mb-2 text-body-secondary">Charles Martin and Jim Torresen. <em>Proceedings of NIME 2019</em>. The original IMPS paper. Introduces the prediction system, focuses on OSC connectivity, and lays out the machine learning approach that later became IMPSY.</p>
    <a href="https://doi.org/10.5281/zenodo.3672952">DOI →</a>
  </li>
  <li>
    <div class="pub-year">2018</div>
    <div class="pub-title">RoboJam: A Musical Mixture Density Network for Collaborative Touchscreen Interaction</div>
    <p class="mb-2 text-body-secondary">Charles Martin and Jim Torresen. <em>Proceedings of EvoMUSArt 2018</em>. An earlier collaborative-performance system that established the MDRNN principle on which IMPS and IMPSY were later built.</p>
    <a href="https://doi.org/10.1007/978-3-319-77583-8_11">DOI →</a>
    · <a href="https://arxiv.org/abs/1711.10746">arXiv →</a>
  </li>
  <li>
    <div class="pub-year">2017</div>
    <div class="pub-title">Deep Models for Ensemble Touch-Screen Improvisation</div>
    <p class="mb-2 text-body-secondary">Charles Martin, Kai Olav Ellefsen, and Jim Torresen. <em>Proceedings of Audio Mostly 2017</em>. Early work applying deep sequence models to musical ensemble interaction; part of the thread leading into RoboJam and IMPS.</p>
    <a href="https://doi.org/10.1145/3123514.3123556">DOI →</a>
  </li>
</ul>

## Work by students and collaborators

These papers extend IMPSY or the MDRNN approach into new instruments, interfaces, and performance contexts.

<ul class="pub-list">
  <li>
    <div class="pub-year">2026</div>
    <div class="pub-title">A Web Interface for Real-Time Interaction with Machine Learning in Musical Performance</div>
    <p class="mb-2 text-body-secondary">Hongdi Zhu and Charles Martin. <em>NIME 2026</em> (to appear). Describes the web interface for IMPSY — configuration, data capture, and model management without specialist tooling.</p>
  </li>
  <li>
    <div class="pub-year">2025</div>
    <div class="pub-title">Touching Wires: Tactility and a Quilted Musical Interface for Human–AI Musical Co-Creation</div>
    <p class="mb-2 text-body-secondary">Sandy Ma and Charles Martin. <em>NIME 2025</em>. A soft, quilted controller for co-creative musical interaction with an IMPSY-based predictive model.</p>
  </li>
  <li>
    <div class="pub-year">2025</div>
    <div class="pub-title">AI See, You See: Human–AI Musical Collaboration in Augmented Reality</div>
    <p class="mb-2 text-body-secondary">Yichen Wang and Charles Martin. <em>CHI EA '25</em>. Uses IMPSY as the predictive model behind a human–AI musical collaboration in head-mounted AR.</p>
    <a href="https://doi.org/10.1145/3706599.3720052">DOI →</a>
  </li>
  <li>
    <div class="pub-year">2024</div>
    <div class="pub-title">Off-the-shelf: Improvising with a Minimal Intelligent Musical Instrument in Mixed Reality</div>
    <p class="mb-2 text-body-secondary">Yichen Wang and Charles Martin. <em>AI Music Creativity (AIMC) 2024</em>. An improvisation study using IMPSY as a minimal predictive partner in a mixed-reality musical setting.</p>
    <a href="https://aimc2024.pubpub.org/pub/ll85912p/release/1">Paper →</a>
    · <a href="https://doi.org/10.5281/zenodo.15110283">Zenodo →</a>
  </li>
</ul>

## Project timeline

<ul class="timeline">
  <li><span class="year">2017–18</span> Started the musical MDRNN idea with the RoboJam project: collaborative touchscreen performance driven by a mixture density network.</li>
  <li><span class="year">2019</span> Released IMPS, generalising the MDRNN approach to arbitrary musical interaction over OSC.</li>
  <li><span class="year">2020</span> Studied IMPS in performance with the EMPI embodied controller; published findings on predictive interaction in <em>Frontiers in AI</em>.</li>
  <li><span class="year">2024</span> Rebuilt IMPS as <strong>IMPSY</strong> with broader I/O, easier configuration, and a focus on Raspberry Pi deployment for new intelligent instruments.</li>
  <li><span class="year">2024</span> Organised the <a href="https://smcclab.au/nime-embedded-ai/">Building NIMEs with Embedded AI</a> workshop at NIME 2024.</li>
  <li><span class="year">2025</span> SMCC Lab students publish the first IMPSY-based papers: Wang (<em>AI See, You See</em>, CHI EA) and Ma (<em>Touching Wires</em>, NIME).</li>
  <li><span class="year">2026</span> Published the IMPSY design-space paper and web-interface paper at NIME 2026; established connections with the <a href="{{ site.links.mishmash }}">Mishmash</a> Centre for AI and Creativity.</li>
</ul>

## Music featuring IMPSY

Recorded music made with IMPSY-based instruments:

- [*Hyphae*](https://collectedresonances.bandcamp.com/album/hyphae) — Collected Resonances.
- [*Clear Skies on a Hill*](https://charlesmartin.bandcamp.com/track/clear-skies-on-a-hill) — Charles Martin.

## Wider context

The IMPSY project is developed at the [Sound, Music, & Creative Computing Lab]({{ site.links.smcclab }}) by [Charles Martin]({{ site.links.charles }}) and collaborators. For ongoing news, talks, and related work, those sites are the best place to look. There is intentionally no blog or news feed here.
