***Models to be used*** <br />
1)No Language Left Behind (NLLB)   [developed by Meta] <br />

The model was trained with input lengths not exceeding 512 tokens, therefore translating longer sequences might result in quality degradation. NLLB-200 translations can not be used as certified translations.
Primary intended uses: NLLB-200 is a machine translation model primarily intended for research in machine translation, - especially for low-resource languages. It allows for single sentence translation among 200 languages.
Information on how to - use the model can be found in Fairseq code repository along with the training code and references to evaluation and training data. <br />
#### Note : Can handle only short input text becoz token size is 500. <br />

Pre-requisites to run NLLB & NLLB code implementation -  https://medium.com/@FaridSharaf/text-translation-using-nllb-and-huggingface-tutorial-7e789e0f7816 <br />
code implementation - https://colab.research.google.com/drive/1fsbzykS5ANEMVcn7gtp8Wl7gkmRzbwOW?usp=sharing


2) Google translator package <br />



#### Steps to perform Text Translation <br />

1) Language detection <br />
   i)detect :libraries to detect language fast-langdetect,langdetect,textblob,lingua. <br />
     1. fast-langdetect or fasttext-langdetect - https://github.com/LlmKira/fast-langdetect  https://pypi.org/project/fasttext-langdetect/ <br />
     2. https://www.geeksforgeeks.org/detect-an-unknown-language-using-python/ <br />
     3. https://medium.com/@monigrancharov/text-language-detection-with-python-beb49d9667b3 <br />
   ii)There are language detection and there probability predicted for it. <br />
   iii)If text contains multiple languages, then use split_lang. Refer link - https://github.com/DoodleBears/split-lang/blob/main/split-lang-demo.ipynb


2) Model implementation

3) 
   



