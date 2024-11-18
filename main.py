from kivymd.app import MDApp
from kivy.uix.screenmanager import Screen, ScreenManager
from kivymd.uix.card import MDCard
from kivy.properties import ObjectProperty, StringProperty, NumericProperty
import datetime
import pytz
# from chatterbot import ChatBot

# import tensorflow as tf
from inference import LLaMA

import numpy as np
import json
from typing import Optional
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from model import ModelArgs, Transformer




def evaluate(sentence, samp_type = 1):

    torch.manual_seed(0)
    model = LLaMA.build(
        checkpoints_dir='llama-2-7b/',
        tokenizer_path='tokenizer.model',
        load_model=True,
        max_seq_len=1024,
        max_batch_size=len(sentence),
        device='cpu'
    )
    out_tokens, out_texts = (model.text_completion(sentence, max_gen_len=64))    
    return out_texts, sentence



class UserInput(MDCard):
    text = StringProperty()
    font_size= NumericProperty()

class BotResponce(MDCard):
    text = StringProperty()
    font_size= NumericProperty()

class ChatScreen(Screen):
    chat_area = ObjectProperty()
    message = ObjectProperty()

    def __init__(self, **kwargs):
        super(ChatScreen, self).__init__(**kwargs)
        # self.chatbot = ChatBot("Donna")

    def send_message(self):
        self.user_input = self.ids.message.text
        self.ids.message.text = ""
        length = len(self.user_input)

        if length >= 40:
            self.ids.chat_area.add_widget(
                UserInput(text=self.user_input, font_size=17, height = length)
            )
        else:
            self.ids.chat_area.add_widget(
                UserInput(text=self.user_input, font_size=17)
            )
        
    def bot_response(self):
        # response = self.chatbot.get_response(self.user_input)
        if self.user_input == "Hello" or self.user_input == "hello" or self.user_input == "hey" or self.user_input == "Hey" or self.user_input == "hi" or self.user_input == "Hi" or self.user_input == "hy" or self.user_input == "Hy" :
            response = "Hello! I'm Donna your personal assistant, how can i help you,"
        elif self.user_input == "what's your name" or self.user_input == "What's your name" or self.user_input == "What is your name" or self.user_input == "what is your name":
            response = "I'm Donna your personal assistant"
        elif self.user_input == "what's the time right now" or self.user_input == "what is the time right now" or self.user_input == "whats the time right now":
            response = "It's ",datetime.datetime.now(pytz.timezone('Asia/Kolkata'))
        else:
            samp_type = 3
            response, sentence = evaluate(self.user_input, samp_type)
        length = len(str(response))

        if length >= 40:
            self.ids.chat_area.add_widget(
                BotResponce(text="{}".format(response), font_size=17, height=length)
            )
        else:
            self.ids.chat_area.add_widget(
                BotResponce(text="{}".format(response), font_size=17)
            )




class ChatApp(MDApp):
    def build(self):
        self.theme_cls.theme_style = "Light"
        self.theme_cls.primary_palette = 'Teal'
        sm = ScreenManager()
        sm.add_widget(ChatScreen(name='chat'))
        return sm
    
if __name__=='__main__':
    ChatApp().run()