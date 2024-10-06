
from django.shortcuts import render,redirect
from django.contrib import messages
from django.shortcuts import render
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from django.contrib.auth import logout




# stockpulseapp/views.py
from django.shortcuts import render
#from .forms import StockForm
# import stockpulseapp.forms as forms
from django.http import HttpResponse
#import django.http as http
# import stockpulseapp.lstm as lstm


# def predictions_view(request):
#     predictions = None  # Initialize predictions variable
#     stock_symbol = None

#     if request.method == 'POST':
#         form = forms.StockForm(request.POST)
#         if form.is_valid():
            
#             stock_symbol = form.cleaned_data['stock_symbol']
#             #stock_symbol = forms.stock_symbol
#             # Call the function in lstm.py to make the prediction
#             #predictions = lstm.make_predictions(forms.stock_symbol)
#             predictions = lstm.predictions
            
            
#     else:
#         form = forms.StockForm()
                          
#     context = {
#         'form': form,
#         'predictions': predictions,
#         'stock_symbol': stock_symbol,
#     }
#     return render(request, 'core/predictions.html', context)







def index(request):
    context = {
        'show_navbar': True,
        'show_footer': True,
    }
    return render(request, 'core/index.html', context)


@login_required
def crypto(request):
    context = {
        'show_navbar': True,
        'show_footer': True,
    }
    return render(request, 'core/cryptocurrency.html', context)

@login_required
def news(request):
    context = {
        'show_navbar': False,
        'show_footer': True,
    }
    return render(request, 'core/news.html', context)

@login_required
def personal(request):
    context = {
        'show_navbar': False,
        'show_footer': False,
    }
    return render(request, 'core/personal.html', context)

@login_required
def calculator(request):
    context = {
        'show_navbar': True,
        'show_footer': True,
    }
    return render(request, 'core/calculator.html', context)

@login_required
def watchlist(request):
    context = {
        'show_navbar': True,
        'show_footer': True,
    }
    return render(request, 'core/watchlist.html', context)

@login_required
def academy(request):
    context = {
        'show_navbar': True,
        'show_footer': False,
    }
    return render(request, 'core/academy.html', context)

@login_required
def streamlit_view(request):
    context = {
        'show_navbar': True,
        'show_footer': True,
    }    
    return render(request, 'core/forecast.html')

def authView(request):
 if request.method == "POST":
  form = UserCreationForm(request.POST or None)
  if form.is_valid():
   form.save()
   return redirect("stockpulseapp:login")
 else:
  form = UserCreationForm()
 return render(request, "registration/signup.html", {"form": form})
    
def logout_view(request):
    logout(request)
    # Redirect to a success page or any other page after logout
    return redirect('stockpulseapp:index')

