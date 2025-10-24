from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),  # trang chủ
    path("api/save-text", views.save_text_request, name="save_text_request"),
    path("api/job-status/<str:job_id>", views.job_status, name="job_status"),
    path("api/single-job", views.single_job_request, name="single_job_request"),
    path("api/save-land-settings", views.save_land_settings, name="save_land_settings"),
]
