import json
import os
from django.db import transaction
from channels.generic.websocket import WebsocketConsumer

from aichat.services import RAGChatService
from rag_app.settings import BASE_DIR
from .models import UserChats
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings


class ChatConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()
    def disconnect(self, close_code):
        pass
    
    def receive(self, text_data):
        text_data_json = json.loads(text_data)
        user_message = text_data_json['message']
        
        if user_message == '':
            
            self.send(
                text_data=json.dumps(
                    {
                        'message': F"I think you forget tell something"
                    }
                )
            )
            
        rag_service = RAGChatService()
        chat_history = []
        
        recent_chats = UserChats.objects.order_by('-created_at')[:3]
        chat_history = [
            {
                "user_message": chat.user_message,
                "ai_message": chat.ai_message
            }
            for chat in reversed(recent_chats)
        ]

        result = rag_service.get_response(user_message, chat_history)
        ai_response = result['answer']
        
        chat = UserChats.objects.create(
            user_message=user_message,
            ai_message=ai_response
        )

        self.send(
            text_data=json.dumps(
                {
                    'message': ai_response
                }
            )
        )
