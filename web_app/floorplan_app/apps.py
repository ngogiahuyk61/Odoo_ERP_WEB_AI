from django.shortcuts import render

def index(request):
    return render(request, 'floorplan_app/index.html')
