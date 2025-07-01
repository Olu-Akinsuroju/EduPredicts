from django.urls import path
from . import views

app_name = 'core'

urlpatterns = [
    # Student Flow
    path('student/', views.student_info, name='student_info'),
    path('student/select-model/', views.student_select_model, name='student_select_model'), # Updated view function
    path('student/predict/', views.student_predict_view, name='student_predict'),

    # Researcher Flow
    path('researcher/', views.researcher_overview_view, name='researcher_overview'),
    path('researcher/upload/', views.researcher_upload_view, name='researcher_upload'),
    path('researcher/results/', views.researcher_results_view, name='researcher_results'),
    path('researcher/download-sample-csv/', views.download_sample_csv, name='download_sample_csv'),
    path('researcher/download-results-csv/', views.download_prediction_results_csv, name='download_prediction_results_csv'),
]
