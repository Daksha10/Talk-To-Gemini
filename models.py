#to find the models available in Google's Generative AI API
import google.generativeai as genai

genai.configure(api_key="AIzaSyDOxeW2OM5Pw9U88t6783h7UBmf0yf2NIg")

models = genai.list_models()
for model in models:
    print(model.name)
