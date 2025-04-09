from django.urls import re_path
from .consumers import ChatConsumer  # Đảm bảo import đúng ChatConsumer

websocket_urlpatterns = [
    re_path(r"ws/chat/$", ChatConsumer.as_asgi()),  # Cấu hình đúng URL
]
