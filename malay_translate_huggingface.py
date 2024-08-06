from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from ftlangdetect import detect
from langcodes import tag_is_valid, Language
import pandas as pd
import spacy
from transformers import MarianMTModel, MarianTokenizer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import LineTokenizer
import math
import torch
import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import malaya
import logging

if torch.cuda.is_available():  
  dev = "cuda"
else:  
  dev = "cpu" 
device = torch.device(dev)

 
# mname = 'Helsinki-NLP/opus-mt-ms-en'
# tokenizer = MarianTokenizer.from_pretrained(mname)
# model = MarianMTModel.from_pretrained(mname)
# model.to(device)
# lt = LineTokenizer()
# batch_size = 8

nlp = spacy.load('en_core_web_sm')

def text_translation(text1,model,tokenizer,model_cd=1,lang_code = 'en'):
    # print(result.values)
    
    if model_cd ==2:
        # model = malaya.translation.huggingface(model = mname)
        this_list = []
        this_list.append(text1)
        # 
        translated_text = model.generate(this_list, to_lang = 'en', max_length = 1000)
        translated_text = "".join(translated_text)
    else:
        # tokenizer = AutoTokenizer.from_pretrained(mname)
        # model = AutoModelForSeq2SeqLM.from_pretrained(mname)
        translator = pipeline('translation', model=model, tokenizer=tokenizer, src_lang=lang_code, tgt_lang='eng_Latn',max_length=len(text1))
        translated_text = translator(text1)
        for key, value in translated_text[0].items():
            # print(key,value)
            if key == 'translation_text':
                translated_text = value
    
    return translated_text
    

def text_preproc(text1):
    
    text1 = text1.replace('\n',' ')
    text1 = text1.replace('\t',' ')
    text1 = text1.replace("..",".").replace(". .",".").replace("\u200c"," ")
    text1 = text1.strip()
    text1 = text1.strip('\n')
    text1 = text1.strip('\t')
    text_new_temp = ''
    text1_split = text1.split('. ')
    if len(text1_split)<3:
        doc = nlp(text1)
        value_list = list(doc.sents)
        text1_split = [str(v) for v in value_list]
    current_text = text1_split[0]
    text_new = ''
    split_list = []
    result = detect(text=text1, low_memory=False)
    for key, value in result.items():
        # print(key,value)
        if key == 'lang':
            text_lang = value 
    if text_lang == "en": 
        return text1

    if tag_is_valid(text_lang):
        print(text_lang)
        lang_name = Language.get(text_lang).display_name("en")
        print(lang_name)
        lang_model = pd.read_excel('/app/Name_Screening/Translation_scraping/Language_Translation/Language_Model.xlsx')
        for i in range(0,lang_model.shape[0]):
            if lang_name==lang_model['V_LANGUAGE_NAME'].iloc[i]:
                model_code = int(lang_model['N_MODEL_CODE'].iloc[i])
                model_name = lang_model['V_MODEL_NAME'].iloc[i]
                break
            else:
                model_code = 1
                model_name = lang_model['V_MODEL_NAME'].iloc[0]
                # print(model_name)
        lang_name_adjusted = lang_name.split('(')[0].strip()
        lang_name_adjusted = " "+lang_name_adjusted

        df_lang = df.loc[df['Language'].str.contains(lang_name_adjusted)]

        if df_lang.shape[0]==0:

            df_lang = df.loc[df['Language'].str.contains(lang_name.split('(')[0].strip())]

        lang_code = df_lang['FLORES-200 code'].iloc[0]

        # print(f"Language name in english: {lang_name,lang_code}")
    else:
        lang_code = 'en'
        # return "Sorry! Could not Translate"
    print(text1_split)
    ########################
    translate_list_list = []
    ########################
    if model_code ==2:
        model = malaya.translation.huggingface(model = model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    for i in range(len(text1_split)):
        # print(i)
        # print(text1_split[i],text1_split[i+1])
        length_max = 250
        if i<len(text1_split)-1 and len(current_text) + len(text1_split[i+1])<length_max:
            current_text = ". ".join([current_text, text1_split[i+1]])
    
        else:
            split_list.append(current_text)
            translate_list_list.append(current_text)
            # print(model_code,lang_code,model_name)
            text_new_temp = text_translation(current_text,model,tokenizer,model_code,lang_code)
            translate_list_list.append(text_new_temp)
            if i<len(text1_split)-1:
                current_text = text1_split[i+1]
    
            if len(text_new)>0:
                text_new = " ".join([text_new, text_new_temp])
            else:
                text_new = text_new_temp
        # print(text_new)
            
            split_list.append(text_new_temp)
    with open("file_list.txt", "w") as output:
        output.write(str(split_list))
    
    # print('final text new')
    file = open('Translated_text.txt', 'w') 
    file.write(str(translate_list_list)) 
    file.close()
    return text_new


# text_new = text_translation(text1)

if __name__ == "__main__":
    import sys

    df_to_be_translated = pd.read_excel("/app/Name_Screening/Translation_scraping/Language_Translation/Translated Output.xlsx")
    df = pd.read_excel('/app/Name_Screening/Translation_scraping/Language_Translation/Language_Mapping_Flora.xlsx')
    for i in range(df_to_be_translated.shape[0]):
        # print(i)
        start_time = time.time()
        text2 = df_to_be_translated["Text"].iloc[i]
        # print(str(df_to_be_translated["Text"].iloc[i]))
        if str(df_to_be_translated["Text"].iloc[i])!='nan' and len(text2)>0 and str(df_to_be_translated["Translated_Text"].iloc[i])=='nan':
            # print(text2)
            text3 = text_preproc(text2)
            # print(text3)
            # pass
        else:
            text3 = df_to_be_translated["Translated_Text"].iloc[i]
        Time_Taken = time.time() - start_time
        # print(text3)
        df_to_be_translated.loc[i,"Translated_Text"] = text3
        df_to_be_translated.loc[i,"Time_Taken"] = Time_Taken
    df_to_be_translated.to_excel("/app/Name_Screening/Translation_scraping/Language_Translation/Translated Output.xlsx",index = False)

