# IMPSY Homepage Repository

This repository holds the homepage source for the IMPSY (Intelligent Musical
Instrument Platform) Project. The source here will build a website that will
serve as a starting point for information about IMPSY, how to install and use
it, and the research projects that have applied IMPSY.

## Contents

The homepage should have a relatively complete front page that explains the concepts behind IMPSY, the basic workflow for creating intelligent musical instruments, and where to find more information.

The information sources should be:

- The main source code repository for IMPSY: <https://github.com/cpmpercussion/impsy>
- The IMPSYpi Raspberry Pi distribution for IMPSY: <https://github.com/cpmpercussion/impsy-pi>
- Videos from this playlist: <https://www.youtube.com/watch?v=GJ6vjhjv-TY&list=PLnRoOVbpGXfa9USZtfxtIkMhU1TfOMKQA>

### Research publications:

These are the main papers by Charles in this project.

- 2026: Opening the Design Space: <https://github.com/cpmpercussion/impsypi-opening-design-space-paper> (most recent paper)
- 2020: Understanding Musical Predictions with EMPI: <https://github.com/cpmpercussion/Understanding-Musical-Predictions-with-EMPI> (paper on embodied musical prediction system)
- 2019: An Interactive Musical Prediction System with Mixture Density Recurrent Neural Networks: <> (original paper introducing the IMPS system, focus on ML and OSC communication)
- 2018: RoboJam: A Musical Mixture Density Network for Collaborative Touchscreen Interaction: <https://github.com/cpmpercussion/robojam-an-interactive-musical-mdn> (early paper, pre-IMPSY, working on MDRNN principle)

### Historical Notes

- 2017-2018: started musical MDRNN idea with Robojam project
- 2019: released IMPS interactive musical prediction system applying MDRNN idea to general musical interaction (focus on OSC connectivity)
- 2020: studied IMPS in performance and with EMPI system
- 2024: rebuilt IMPS as IMPSY with broad IO connectivity and easier configurability. Focus on running on Rasbperry Pi for developing many new intelligent instruments.
- 2024: presented IMPSYpi in workshop at NIME2024: <https://github.com/smcclab/nime-embedded-ai>
- 2026: updated IMPSY for modern install and clarifying the web interface. Establishing connections with the Mishmash project <https://mishmash.no>

## Style

- The style should be creative yet research oriented, typical of the NIME community.
- For examples of style, consulte the research publication above for Charles Martin's voice.
- The goal is not marketing copy, but clear communication to fellow researchers and music technologists who might adopt and adapt this work.
- For further information about style and layout see Charles website: <https://github.com/cpmpercussion/homepage>
- Few pages with clear markdown style for later updating.
- no need for a blog or news. Link to Charles Martin's website or the SMCClab site <https://smcclab.au>
- clear minimal visual design with light/dark/auto switching.
- visual elements driven by actual performance and system images (see paper repositories) not by stock images or plain generated graphics.

## Technical details

- Jekyll website
- designed to build to github pages, needs a github action
- should use bootstrap for style and a custom SCSS file somewhere.
