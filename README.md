


## Related works

- Kim 2014: Convolutional Neural Networks for Sentence Classification https://www.aclweb.org/anthology/D14-1181/

- Jafariakinabad et al. 2019: Style-aware Neural Model with Application in Authorship Attribution https://arxiv.org/abs/1909.06194 (CNN + LSTM + Attention)

- Mohsen et al. 2016: Author Identification Using Deep Learning https://ieeexplore.ieee.org/document/7838265 (stacked denoising autoencoder + SVM classifier)

- Mir et al. 2019: Evaluating Style Transfer for Text https://arxiv.org/abs/1904.02295

- Rocha et al. 2015: Authorship attribution for social media forensics https://ieeexplore.ieee.org/abstract/document/7555393

- Ruder et al. 2016: Character-level and Multi-channel Convolutional Neural Networks for Large-scale Authorship Attribution https://arxiv.org/abs/1609.06686

- Sapkota et al. 2015: Not All Character N-grams Are Created Equal: A Study in Authorship Attribution https://www.aclweb.org/anthology/N15-1010/

- Sari et al. 2017: Continuous N-gram Representations for Authorship Attribution  https://www.aclweb.org/anthology/E17-2043/

- Shrestha et al. 2017: Convolutional Neural Networks for Authorship Attribution of Short Texts https://www.aclweb.org/anthology/E17-2106/

- Bownman et al. 2016, Generating Sentences from a Contiuous Space https://arxiv.org/pdf/1511.06349.pdf

- Sanh et al. 2019: DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter https://arxiv.org/pdf/1910.01108.pdf


## Available model structures & repos:

- GPT2: https://github.com/ConnorJL/GPT2

- Text-CNN: 
  - https://github.com/dennybritz/cnn-text-classification-tf
  - https://github.com/Shawn1993/cnn-text-classification-pytorch
  - https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras
  - https://github.com/DongjunLee/text-cnn-tensorflow

- FastText: https://github.com/facebookresearch/fastText

- Pre-trained GloVe: https://nlp.stanford.edu/projects/glove/

- Hand-build models: 
  - CNN + GRU w/ skip connections
  - CNN + GRU + Attention w/ skip connections
  - CNN + GRU + Doc2Vec + Attension w/ skip connections
  
- GLTR: http://gltr.io/

- Source for adversarial classifiers (CAAE, ARAE) for classifying naturalness - https://arxiv.org/pdf/1511.06349.pdf

- Stanford NER for discriminator: https://nlp.stanford.edu/software/CRF-NER.html

## To-dos:

- Talk to Kian / Andrew for further guidance
- Closing the loop: use discriminator as an extra filter to select the best K examples, and then use them as input to generator?

- For the upcoming week (from 11.13):
  - NER checking
  - Include interprebility for discriminators (e.g. highest activation, evaluation metrics)
  - Paper editing on overleaf: get overall scores; fill two-line classifier architecture into paper (look for TBD comments)
  - scope out MTurk 
  - Read examples of good projects posted
