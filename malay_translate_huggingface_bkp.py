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

def text_translation(text1,model=1,lang_code = 'en',mname='facebook/nllb-200-distilled-1.3B'):
    # print(result.values)
    
    # if model == 1:
        # tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B")
        # model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-1.3B")
        # translator = pipeline('translation', model=model, tokenizer=tokenizer, src_lang=lang_code, tgt_lang='eng_Latn',max_length=len(text1))
        # translated_text = translator(text1)
        # for key, value in translated_text[0].items():
            # print(key,value)
            # if key == 'translation_text':
                # translated_text = value
    if model ==2:
        model = malaya.translation.huggingface(model = mname)
        print(text1,'\n')
        this_list = []
        this_list.append(text1)
        # 
        translated_text = model.generate(this_list, to_lang = 'en', max_length = 1000)
        translated_text = "".join(translated_text)
        print(translated_text,'\n')
        
    else:
        if model == 3:
            mname = 'Helsinki-NLP/opus-mt-vi-en'
        elif model ==4:
            mname = 'Helsinki-NLP/opus-mt-zh-en'
        elif model ==5:
            mname = 'Helsinki-NLP/opus-mt-tc-big-fr-en'
        else:
            mname = 'facebook/nllb-200-distilled-1.3B'
        
        
        tokenizer = AutoTokenizer.from_pretrained(mname)
        model = AutoModelForSeq2SeqLM.from_pretrained(mname)
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
            else:
                model_code = 1
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
    for i in range(len(text1_split)):
        # print(i)
        # print(text1_split[i],text1_split[i+1])
        length_max = 50
        if i<len(text1_split)-1 and len(current_text) + len(text1_split[i+1])<length_max:
            current_text = ". ".join([current_text, text1_split[i+1]])
    
        else:
            split_list.append(current_text)
            translate_list_list.append(current_text)
            text_new_temp = text_translation(current_text,model_code,lang_code,model_name)
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
    # text2 = " ఐపీఎల్ 2024 సీజన్‌‌ ముగింపుకొచ్చేసింది. లీగ్స్ దశలో ఇంకొన్ని మ్యాచ్‌లే మిగిలివున్నాయి. ఢిల్లీ కేపిటల్స్ మినహా మిగిలినవన్నీ కూడా తమ చివరి మ్యాచ్‌లను ఆడుతున్నాయి. ఢిల్లీ కేపిటల్స్ తన చిట్టచివరి మ్యాచ్‌ను ఆడేసింది. లీగ్స్‌ దశలో అన్ని మ్యాచ్‌లనూ ఆడేసిన మొట్టమొదటి జట్టు అదే. తన సొంత గడ్డపై జరిగిన మ్యాచ్‌లో లక్నో సూపర్ జెయింట్స్‌పై విజయదుందుభి మోగించింది. గెలుపుతో ఈ సీజన్‌కు ముగింపు పలికింది. మంగళవారం రాత్రి ఢిల్లీ అరుణ్ జైట్లీ స్టేడియంలో జరిగిన ఈ మ్యాచ్‌లో లక్నో సూపర్ జెయింట్స్‌ను 19 పరుగుల తేడాతో మట్టికరిపించింది రిషభ్ సేన. తొలుత బ్యాటింగ్‌కు నిర్ణీత 20 ఓవర్లల్లో నాలుగు వికెట్ల నష్టానికి 208 పరుగులు చేసింది. అనంతరం లక్నోను 189 పరుగులకు కట్టడి చేయగలిగింది. ఢిల్లీ కేపిటల్స్ సాధించిన ఈ గెలుపు రాయల్ ఛాలెంజర్స్ బెంగళూరు ప్లేఆఫ్స్ ఆశలను గండికొట్టినట్టయింది. ప్రస్తుతం ఆర్సీబీ ఖాతాలో 12 పాయింట్లు ఉన్నాయి. ఇంకో మ్యాచ్ మిగిలే వుంది. చెన్నై సూపర్ కింగ్స్‌తో తలపడాల్సి ఉంది. ఇందులో ఓడితే ఆర్సీబీ ప్రస్థానం 12 పాయింట్ల వద్దే స్తంభించిపోతుంది. గెలిస్తే మాత్రం ఆసక్తికరంగా మారతాయి ఆర్సీబీ ప్లేఆఫ్స్ అవకాశాలు. చెన్నై సూపర్ కింగ్స్‌పై గెలిస్తే ఆర్సీబీ పాయింట్లు కూడా 14కు చేరుకుంటాయి. పాయింట్లల్లో ఢిల్లీ కేపిటల్స్‌తో సమానంగా నిలుస్తుంది. దాని నెట్ రన్‌రేట్‌ను అధిగమించేలా ఆర్సీబీ తన ప్రత్యర్థి చెన్నై సూపర్ కింగ్స్‌ను ఓడించాల్సి ఉంటుంది. ఢిల్లీ కేపిటల్స్ నెట్ రన్‌రేట్ మైనస్ 0.377. ఆర్సీబీ నెట్ రన్‌రేట్ ప్లస్ 0.387. ఆర్సీబీ తొలుత బ్యాటింగ్ చేస్తే 18 పరుగుల తేడాతో చెన్నై సూపర్ కింగ్స్‌ను ఓడించాల్సి ఉంటుంది. లేదా.. చెన్నై నిర్దేశించే లక్ష్యాన్ని 18.1 ఓవర్లల్లో ఛేదించాల్సి ఉంటుంది. అప్పుడే నెట్ రన్‌రేట్ మెరుగుపడుతుంది. ఎటు తిరిగీ ఆర్సీబీ.. చెన్నై సూపర్ కింగ్స్‌పై గెలవాల్సి ఉంటుంది"
    df_to_be_translated = pd.read_excel("/app/Name_Screening/Translation_scraping/Language_Translation/Translated Output.xlsx")
    df = pd.read_excel('/app/Name_Screening/Translation_scraping/Language_Translation/Language_Mapping_Flora.xlsx')
    for i in range(df_to_be_translated.shape[0]):
        print(i)
        text2 = df_to_be_translated["Text"].iloc[i]
        # print(str(df_to_be_translated["Text"].iloc[i]))
        if str(df_to_be_translated["Text"].iloc[i])!='nan' and len(text2)>0 and str(df_to_be_translated["Translated_Text"].iloc[i])=='nan':
            # print(text2)
            text3 = text_preproc(text2)
            # print(text3)
            # pass
        else:
            text3 = df_to_be_translated["Translated_Text"].iloc[i]
        
        # print(text3)
        df_to_be_translated.loc[i,"Translated_Text"] = text3
    df_to_be_translated.to_excel("/app/Name_Screening/Translation_scraping/Language_Translation/Translated Output.xlsx",index = False)

