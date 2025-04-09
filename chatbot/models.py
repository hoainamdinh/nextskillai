from django.db import models
import random, datetime
from django.utils import timezone
import pandas as pd
from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.urls import reverse

def generate_id(cls, field_name, prefix, num_length): 
    """Tạo ID tự động với prefix và độ dài cố định"""        
    while True:
        num_part = str(random.randint(0, 10**num_length - 1)).zfill(num_length)
        new_id = prefix + num_part
        if not cls.objects.filter(**{field_name: new_id}).exists():
            return new_id

