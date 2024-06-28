# NLP machine learning sample scripts
- `uni_ner.py` contains a quick implementation of NER using a fine-tuned Llama 2 model
  - Code based on https://github.com/universal-ner/universal-ner and the paper https://arxiv.org/abs/2308.03279
  - Related: `uni_ner_eval.ipynb`, a quick notebook for evaluating the Universal NER model vs SpaCy's NER and human annotations
- `phi2_finetune.ipynb`, showing fine-tuning of the Phi-2 language model, pushing to HuggingFace, and running it after
- `HF_distilbert_classifier.ipynb`, showing a quick fine-tune for a distilBERT text classifier
