
# stockpulseapp/forms.py
from django import forms

class StockForm(forms.Form):
    stock_symbol = forms.CharField(label='Stock Symbol', max_length=10, widget=forms.TextInput(attrs={'placeholder': 'Enter stock symbol'}))
