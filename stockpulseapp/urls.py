from django.urls import path,include
import stockpulseapp.views as views
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.auth import views as auth_views
#from .views import authView, index,logout_view




urlpatterns = [
    
    # path('predictions/', views.predictions_view, name='predictions'),
    
    path('', views.index, name='index'),
    path('crypto/', views.crypto, name='crypto'),
    path('news/', views.news, name='news'),
    path('watchlist/', views.watchlist, name='watchlist'),
    path('personal/', views.personal, name='personal'),
    path('academy/', views.academy, name='academy'),
    path('calculator/', views.calculator, name='calculator'),
    path('forecast/', views.streamlit_view, name='forecast'),
    path("signup/", views.authView, name="authView"),
    path("accounts/", include("django.contrib.auth.urls")),
    path('logout/', views.logout_view, name='logout'),
    
]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    