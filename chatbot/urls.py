from django.urls import path, include
from . import views

app_name = 'chatbot'
## module san pham
urlpatterns = [
    path('', views.index, name='index'),
]


from django.urls import path 
from . import views 

urlpatterns += [
    path('chatapp/', views.chat_view , name='chat_view'),
]


