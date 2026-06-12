---
layout: home
title: "IMPSY: Intelligent Musical Instrument Platform"
description: A research toolkit for building intelligent musical instruments with mixture density recurrent neural networks. Runs on Raspberry Pi, talks OSC, MIDI, serial, and the web.
permalink: /
image: /assets/img/impsy-s1-soundout.jpg
---

<section class="hero">
  <div class="container">
    <div class="row align-items-center g-5">
      <div class="col-lg-6">
        <p class="hero-eyebrow mb-3">Intelligent Musical Instrument Platform</p>
        <h1 class="display-4 mb-3">Instruments that listen, predict, and play along.</h1>
        <p class="hero-lede mb-4">
          IMPSY is a research toolkit for building intelligent musical instruments using
          mixture density recurrent neural networks. Train it on your own performance data,
          run it on a Raspberry Pi, and connect it to almost anything that speaks OSC, MIDI,
          serial, or the web.
        </p>
        <div class="d-flex flex-wrap gap-2">
          <a class="btn btn-impsy btn-lg" href="{{ '/get-started/' | relative_url }}">Get started</a>
          <a class="btn btn-outline-secondary btn-lg" href="{{ '/workshop/' | relative_url }}">IMPSYpi workshop</a>
          <a class="btn btn-outline-secondary btn-lg" href="{{ site.links.source }}">Source on GitHub</a>
          <a class="btn btn-outline-secondary btn-lg" href="{{ '/research/' | relative_url }}">Read the research</a>
        </div>
      </div>
      <div class="col-lg-6 hero-figure">
        <img src="{{ '/assets/img/impsy-s-1-169.jpg' | relative_url }}"
             alt="An IMPSY-driven Korg S-1 synthesizer in performance, with a Raspberry Pi controller running the predictive model."
             loading="eager">
      </div>
    </div>
  </div>
</section>

<section class="container py-5">
  <div class="row mb-4">
    <div class="col-lg-8">
      <h2 class="fw-semibold mb-3">What IMPSY does</h2>
      <p class="lead text-body-secondary mb-0">
        IMPSY captures a stream of musical gestures from a controller, sensor, or touchscreen,
        learns the temporal shape of a performer's choices, and generates plausible continuations in real time.
        It can run embedded inside an instrument, or as a separate module connected to one.
      </p>
    </div>
  </div>

  <div class="row g-4">
    <div class="col-md-4">
      <div class="workflow-step">
        <span class="step-num">01 / SENSE</span>
        <h3>Capture gestures</h3>
        <p>Stream multi-dimensional control data into IMPSY over OSC, MIDI, serial, or websockets, whichever your instrument speaks.</p>
      </div>
    </div>
    <div class="col-md-4">
      <div class="workflow-step">
        <span class="step-num">02 / LEARN</span>
        <h3>Train an MDRNN</h3>
        <p>A mixture density recurrent network learns the joint distribution of your gestures over time, on a laptop or a Pi.</p>
      </div>
    </div>
    <div class="col-md-4">
      <div class="workflow-step">
        <span class="step-num">03 / PREDICT</span>
        <h3>Play together</h3>
        <p>IMPSY generates continuations or accompaniments at performance latency, routed back into your instrument's sound engine.</p>
      </div>
    </div>
  </div>
</section>

<section class="container py-5">
  <div class="row align-items-center g-5">
    <div class="col-md-6">
      <img class="img-fluid rounded shadow-sm"
           src="{{ '/assets/img/impsy-diagram.png' | relative_url }}"
           alt="Block diagram of the IMPSY system: input adapters, mixture density RNN, output routing.">
    </div>
    <div class="col-md-6">
      <h2 class="fw-semibold">A small system, designed to be opened</h2>
      <p>
        IMPSY is a Python package with a small footprint and clear configuration. Inputs and outputs are decoupled
        from the model, so the same trained network can drive a synthesizer, an iPad app, or a hardware controller
        without re-training. A web interface exposes recording, training, and inference.
      </p>
      <p class="mb-0">
        The companion <a href="{{ site.links.pi }}">IMPSYpi</a> distribution packages the toolkit for Raspberry Pi Zero 2 W, 3, 4, and 5 so that an intelligent instrument can run untethered on stage. The <a href="{{ '/workshop/' | relative_url }}">IMPSYpi workshop</a> walks through the whole process end to end.
      </p>
    </div>
  </div>
</section>

<section class="container py-5">
  <div class="row mb-4">
    <div class="col-lg-8">
      <h2 class="fw-semibold mb-3">Instruments built with IMPSY</h2>
      <p class="text-body-secondary">
        A short tour of instruments we've built with IMPSY.
        Most of these images come from the <a href="https://github.com/cpmpercussion/impsypi-opening-design-space-paper">2026 design-space paper</a>.
      </p>
    </div>
  </div>

  <div class="row g-3 gallery-grid">
    <div class="col-6 col-md-3">
      <img src="{{ '/assets/img/intelligent-s1.jpg' | relative_url }}" alt="Korg S-1 with IMPSYpi controller.">
    </div>
    <div class="col-6 col-md-3">
      <img src="{{ '/assets/img/intelligent-microfreak.jpg' | relative_url }}" alt="Arturia MicroFreak driven by IMPSY.">
    </div>
    <div class="col-6 col-md-3">
      <img src="{{ '/assets/img/ipad-demo-controller.jpg' | relative_url }}" alt="iPad-based IMPSY controller.">
    </div>
    <div class="col-6 col-md-3">
      <img src="{{ '/assets/img/garage-concert.jpg' | relative_url }}" alt="An IMPSY-driven instrument performing live.">
    </div>
  </div>

  <div class="text-center mt-4">
    <a class="btn btn-outline-secondary" href="{{ '/gallery/' | relative_url }}">More instruments &amp; videos →</a>
  </div>
</section>

<section class="container py-5">
  <div class="row">
    <div class="col-md-6">
      <h2 class="fw-semibold">Read the research</h2>
      <p>IMPSY grew out of a sequence of research papers on mixture density networks for musical interaction, embodied prediction, and the design space of intelligent instruments.</p>
      <a class="btn btn-outline-secondary" href="{{ '/research/' | relative_url }}">Publications →</a>
    </div>
    <div class="col-md-6">
      <h2 class="fw-semibold">Get involved</h2>
      <p>
        IMPSY is open source and developed in the open. The core repository, the Pi distribution,
        and this site all live on GitHub — issues and pull requests are welcome.
      </p>
      <a class="btn btn-outline-secondary" href="{{ site.links.source }}">Star on GitHub →</a>
    </div>
  </div>
</section>
