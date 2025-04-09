from django.urls import path
from . import views

urlpatterns = [
    path('', views.homeView, name='home'),
    path('signatureVerification/', views.signatureVerificationView, name='signatureVerification'),
]