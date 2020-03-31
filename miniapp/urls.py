from django.urls import path
from . import views

urlpatterns = [
    path('', views.title, name='title_name'),
    path('nb',views.nb,name='nb'),
    path('rf',views.rf,name='rf'),
    path('lr',views.lr,name='lr'),
]