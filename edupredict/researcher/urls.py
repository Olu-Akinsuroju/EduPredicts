from django.urls import path
from . import views

app_name = 'researcher'

urlpatterns = [
    path('overview/', views.overview, name='overview'),
]
