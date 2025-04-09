from time import sleep
from django.core.mail import send_mail
from django import forms
from .tasks import send_feedback_email_task
