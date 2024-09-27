# Llama3-Anomaly-Detection

This repository holds the code for implementation of different techniques to detect anomalies present in the device sizes data of NIH medical devices (downloaded from AccessGUDID website)

Note: All these notebooks were run using Google Colab. You need deviceSizes.txt and gmdnTerms.txt files from the AccessGUDID website to test these notebooks.
```
1. anom_det_classical.ipynb
```
Classical implementation of anomaly detection which uses SI unit normalization and classical vectorization to create a vectorized dataset. On top of this vectorized dataset, classical ML based anomaly detection algorithm isoforests is run to detect anomalies
```
2. anom_det_sBeRT.ipynb
```
Similar implementation, but the vectorization process is replaced using a transformer based encoder - Vanilla BERT, sBERT light and sBERT heavy were tested. This is followed by performing anomaly detection on the vectorized dataset using isoforests.
```
3. NIH_LLAMA3_02_FINE_TUNE.ipynb
```
This involves the fine-tuning of Llama3-8B to process the device sizes data intelligently such that classical vectorization would suffice eliminating the need for BERT based computationally heavy vectorization
