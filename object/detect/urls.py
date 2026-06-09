from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("about/", views.about, name="about"),
    path("contact/", views.contact, name="contact"),
    path("api/detect/", views.api_detect, name="api_detect"),
    path('history/', views.detection_history, name='detection_history'),
]