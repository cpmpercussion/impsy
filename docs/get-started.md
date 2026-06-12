---
layout: page
title: Get Started
subtitle: From a fresh checkout to an instrument that predicts.
permalink: /get-started/
description: How to install IMPSY, capture gesture data, train a mixture density RNN, and route predictions back into your instrument.
---

IMPSY is a Python project. There are three main paths: a **desktop install** for development, training, and recording; the **IMPSYpi** distribution for running an intelligent instrument on a Raspberry Pi; and the **[IMPSY AUv3 app]({{ site.links.auv3 }})** for playing IMPSY models inside a DAW (or standalone) on iOS and macOS.

> **Looking for the most complete walkthrough?** The [IMPSYpi Workshop]({{ '/workshop/' | relative_url }}) takes you end-to-end: Docker install on a laptop, training your own MDRNN, flashing the Raspberry Pi image, and connecting IMPSY to a micro:bit or Pure Data instrument. The page below is the short version — if you want the full guided tour, start there.

## 1. Install IMPSY on your computer

Clone the [main repository]({{ site.links.source }}) and follow the install instructions there. IMPSY uses a standard Python toolchain and should work on macOS, Linux, and Windows.

```bash
git clone https://github.com/cpmpercussion/impsy.git
cd impsy
# follow the README for the current install steps
```

The repository README is the source of truth for environment setup, dependency versions, and CLI commands — this page links to it rather than duplicating it.

## 2. Choose your I/O

IMPSY treats input and output as separate concerns from the model. You configure adapters for whatever your instrument speaks:

- **OSC**: the original and most flexible path; works with Pure Data, Max, SuperCollider, TouchOSC, and most DAWs via plugins.
- **MIDI**: for connecting to off-the-shelf synths and controllers.
- **Serial**: for microcontrollers and custom hardware (sensors, touch surfaces, gesture devices).
- **Web**: a built-in interface for recording, training, and live performance from a browser.

A single trained model can be driven by any of these without retraining.

## 3. Record some gestures

The basic workflow is to **record yourself playing first**. IMPSY learns the temporal shape of *your* performance choices, and the model is small enough that a few minutes of focused recording is often enough to start producing useful predictions.

The web interface makes this straightforward: hit record, perform, save the log. The CLI offers the same with more control over file paths and dimensions.

## 4. Train a mixture density RNN

IMPSY's prediction model is a small mixture density recurrent network. Training runs locally: no cloud component, no account, no required GPU for modestly sized models. Training a usable instrument on a few minutes of data takes minutes, not hours.

See the main repository for the current training command, hyperparameter defaults, and tips for tuning the mixture parameters.

## 5. Run the instrument

Once you have a trained model, IMPSY can run in *call-and-response* mode (the model answers when you pause), *polyphony* mode (you and the model play simultaneously), or *battle* mode (the model plays autonomously alongside you) — plus a *user-only* mode for logging data without predictions. See the [configuration reference]({{ '/config/' | relative_url }}) for details. On a desktop machine you can iterate quickly; on a Pi, you get a self-contained intelligent instrument that boots straight into performance mode.

## Going embedded: IMPSYpi

[IMPSYpi]({{ site.links.pi }}) is a Raspberry Pi distribution that ships IMPSY pre-configured for embedded use: it boots quickly, exposes the web interface on the local network, and is designed to live inside an instrument enclosure. It targets Raspberry Pi 4 and 5 (Pi 3 and Zero 2 W also work).

The IMPSYpi repository contains the build scripts, image notes, and example deployments — start there if your goal is a standalone, untethered intelligent instrument.

## In your DAW: the IMPSY AUv3 app

[IMPSY AUv3]({{ site.links.auv3 }}) is a native iOS/macOS app that runs IMPSY's TFLite models as an AUv3 MIDI processor — load it in AUM, Logic Pro, or any other AUv3 host, or run it standalone with virtual MIDI ports. It plays in call-and-response mode, imports and exports the same TOML mappings as the Python toolkit, and can record session logs in IMPSY's `.log` format ready for the training pipeline, so models and data move freely between the app and a desktop install.

It's [on the App Store](https://apps.apple.com/app/id6771762122) for iOS 17+ and macOS 14+ (Apple Silicon); the source lives at [cpmpercussion/impsy-auv3](https://github.com/cpmpercussion/impsy-auv3).

## Watch it in action

The [IMPSY video playlist on YouTube]({{ site.links.videos }}) shows the system being trained, performed with, and embedded in a range of instruments. The [2026 paper](https://github.com/cpmpercussion/impsypi-opening-design-space-paper) walks through the design space in more depth.

## Where to ask questions

- **Bugs and feature requests** — open an [issue on GitHub]({{ site.links.source }}/issues)
- **Research collaborations** — see the [SMCC Lab]({{ site.links.smcclab }}) site or [Charles Martin's homepage]({{ site.links.charles }})
