
from django.urls import re_path
from rag_app.consumers import ChatConsumer


websocket_urlpatterns = [
    re_path(r'^chat/$', ChatConsumer.as_asgi()),
]