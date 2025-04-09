from django.shortcuts import render, redirect, get_object_or_404
from . import models
import pandas as pd
from django.contrib import messages
from django.http import JsonResponse
import json
import gspread
from oauth2client.service_account import ServiceAccountCredentials

from rest_framework.response import Response
from rest_framework.decorators import api_view

# Create your views here.
def index(request):
    return render(request, 'index.html')


def chat_view(request):
    return render(request, "chat/chat.html")
