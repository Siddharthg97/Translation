***Models to be used***
1)NLLB <br />

The model was trained with input lengths not exceeding 512 tokens, therefore translating longer sequences might result in quality degradation. NLLB-200 translations can not be used as certified translations.
Primary intended uses: NLLB-200 is a machine translation model primarily intended for research in machine translation, - especially for low-resource languages. It allows for single sentence translation among 200 languages.
Information on how to - use the model can be found in Fairseq code repository along with the training code and references to evaluation and training data. <br />
#### Note : Can handle only short input text becoz token size is 500. 
