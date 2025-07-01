from django.shortcuts import render, redirect
from django.urls import reverse
from django.http import HttpResponse, FileResponse
from .forms import StudentInfoForm, ModelSelectionForm, FileUploadForm
from .ml_utils import predict_student, batch_predict, get_sample_csv_data
import pandas as pd
import io
import os
from django.conf import settings
from django.contrib import messages # For displaying messages

# Landing Page
def landing_page(request):
    return render(request, 'landing.html')

# Student Flow
def student_info(request): # Renamed from student_form_view, new template
    if request.method == 'POST':
        form = StudentInfoForm(request.POST)
        if form.is_valid():
            # Convert form data to a dictionary that is JSON serializable for the session
            # For ChoiceFields, cleaned_data will give the 'key' (e.g., 1, 2, 3).
            # For BooleanFields, it will be True/False.
            # This is generally fine for session storage.
            student_data_for_session = {}
            for key, value in form.cleaned_data.items():
                student_data_for_session[key] = value

            request.session['student_data'] = student_data_for_session
            messages.success(request, "Step 1 complete. Please select a model.")
            return redirect(reverse('core:student_select_model'))
        else:
            messages.error(request, "Please correct the errors below.")
    else:
        form = StudentInfoForm()
    return render(request, 'student/form.html', {'form': form}) # Changed template path

def student_select_model_view(request):
    student_data = request.session.get('student_data')
    if not student_data:
        messages.error(request, "Please submit your information first.")
        return redirect(reverse('core:student_info')) # Redirect to the new student_info view

    if request.method == 'POST':
        form = ModelSelectionForm(request.POST)
        if form.is_valid():
            request.session['model_name'] = form.cleaned_data['model_name']
            # The form in student_select_model.html should POST to student_predict_view
            # So, this view prepares data, and the actual prediction happens in student_predict_view
            # which is called by the form submission from student_select_model.html.
            # Thus, we redirect to student_predict view which will handle the actual prediction
            # based on the data now in session.
            # Correction: The form in select_model.html should POST to student_predict.
            # This view (student_select_model_view) is for GET display of the form,
            # and if it receives a POST, it means the model was selected and it should
            # prepare for the actual prediction call.
            # The most straightforward way is to have student_select_model.html's form
            # POST to student_predict. This view then is only for rendering the selection form.
            # Let's adjust: student_select_model.html's form will POST to student_predict.
            # This view just renders the form.
            # No, the original plan was:
            # Step 2: On POST, validate and redirect to /student/select-model/, where a user picks one of three radio options (Logistic, Tree, RF).
            # Step 3: On POST to /student/predict/, call our existing ML pipeline
            # This means student_select_model_view's form should POST to student_predict.

            # Storing model in session and redirecting to predict view (which will be GET)
            # This is one way, student_predict then reads from session.
             return redirect(reverse('core:student_predict'))
    else:
        form = ModelSelectionForm()

    return render(request, 'core/student_select_model.html', {'form': form})


def student_predict_view(request):
    student_data = request.session.get('student_data')
    model_name = request.session.get('model_name')

    if not student_data:
        messages.error(request, "Student data not found. Please start over.")
        return redirect(reverse('core:student_form'))
    if not model_name:
        messages.error(request, "Model not selected. Please select a model.")
        return redirect(reverse('core:student_select_model'))

    # This view is now accessed via GET after model_name is set in session by student_select_model_view's POST handler
    prediction_result = predict_student(student_data, model_name)

    # We can keep student_data and model_name in session if we want the user to be able
    # to go back to select_model and pick a different model for the *same* data.
    # Or clear them to force a full restart. Let's keep them for now.
    # del request.session['student_data']
    # del request.session['model_name']

    return render(request, 'core/student_result.html', {
        'result': prediction_result,
        'student_data': student_data,
        'model_name': model_name
    })


# Researcher Flow
def researcher_overview_view(request):
    # Provide a link/button to download sample_student_data.csv
    # The actual download will be handled by a separate view if needed, or a static file link.
    # For now, this view just renders the template.
    # The template will contain the link: {% url 'core:download_sample_csv' %}
    return render(request, 'core/researcher_overview.html')

def download_sample_csv(request):
    # This view is linked from researcher_overview.html
    csv_data = get_sample_csv_data()
    response = HttpResponse(csv_data, content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="sample_student_data.csv"'
    return response

def researcher_upload_view(request):
    if request.method == 'POST':
        form = FileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES['file']

            try:
                # Store the uploaded file's content in session.
                # This is okay for small files. For large files, saving to a temp file is better.
                # Let's use a temporary file path stored in session.

                # Ensure temp directory exists
                temp_dir = os.path.join(settings.BASE_DIR, 'tmp')
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)

                # Use a more unique temp file name if multiple users could access this simultaneously
                # For this example, a fixed name is fine, but in production, use tempfile.NamedTemporaryFile
                temp_file_path = os.path.join(temp_dir, 'uploaded_batch_data.csv')

                with open(temp_file_path, 'wb+') as destination:
                    for chunk in uploaded_file.chunks():
                        destination.write(chunk)

                request.session['uploaded_file_path'] = temp_file_path

                # Preview: Read from the saved temp file
                df_preview = pd.read_csv(temp_file_path, nrows=5)
                preview_html = df_preview.to_html(classes=['min-w-full', 'divide-y', 'divide-gray-200', 'border', 'border-gray-300', 'table-auto'], border=0, justify='left', index=False)

                return render(request, 'core/researcher_upload.html', {
                    'form': form,
                    'preview_html': preview_html,
                    'file_uploaded': True
                })

            except Exception as e:
                messages.error(request, f"Error processing file: {e}")
                if 'uploaded_file_path' in request.session:
                    if os.path.exists(request.session['uploaded_file_path']):
                        os.remove(request.session['uploaded_file_path'])
                    del request.session['uploaded_file_path']
                return render(request, 'core/researcher_upload.html', {'form': form, 'preview_html': None, 'file_uploaded': False})
    else:
        form = FileUploadForm()
        # Clean up any stale temp file if the user navigates here via GET
        stale_file_path = request.session.pop('uploaded_file_path', None)
        if stale_file_path and os.path.exists(stale_file_path):
            os.remove(stale_file_path)

        stale_results_path = request.session.pop('results_file_path', None)
        if stale_results_path and os.path.exists(stale_results_path):
            os.remove(stale_results_path)
        if 'results_csv_filename' in request.session:
            del request.session['results_csv_filename']


    return render(request, 'core/researcher_upload.html', {'form': form, 'preview_html': None, 'file_uploaded': False})

def researcher_results_view(request):
    # This view should be POSTed to from the "Run Predictions" button on researcher_upload.html
    if request.method == 'POST':
        file_path = request.session.get('uploaded_file_path')

        if not file_path or not os.path.exists(file_path):
            messages.error(request, "Uploaded file not found. Please upload again.")
            return redirect(reverse('core:researcher_upload'))

        try:
            results_df = batch_predict(file_path) # batch_predict reads from file_path

            # Prepare results for display (e.g., top 50 rows)
            results_preview_html = results_df.head(50).to_html(classes=['min-w-full', 'divide-y', 'divide-gray-200', 'border', 'border-gray-300', 'table-auto'], border=0, justify='left', index=False)

            # Save the full results DF to a new temporary CSV file for download
            temp_dir = os.path.join(settings.BASE_DIR, 'tmp')
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

            results_filename = "prediction_results.csv"
            results_file_path = os.path.join(temp_dir, results_filename)
            results_df.to_csv(results_file_path, index=False)

            request.session['results_file_path'] = results_file_path # Store path for download view
            request.session['results_csv_filename'] = results_filename # Store filename for download view

            # Clean up the original uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)
            if 'uploaded_file_path' in request.session: # should be redundant due to pop earlier, but good practice
                request.session.pop('uploaded_file_path', None)

            return render(request, 'core/researcher_results.html', {
                'results_html': results_preview_html,
                'download_ready': True,
                'results_filename': results_filename
            })
        except Exception as e:
            messages.error(request, f"Error during batch prediction: {e}")
            # Clean up uploaded file in case of error
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
            request.session.pop('uploaded_file_path', None)
            return redirect(reverse('core:researcher_upload'))
    else:
        # GET request to this URL usually means direct access or error, redirect.
        messages.info(request, "To see results, please upload a file and run predictions.")
        return redirect(reverse('core:researcher_upload'))


def download_prediction_results_csv(request):
    results_file_path = request.session.get('results_file_path')
    results_filename = request.session.get('results_csv_filename', 'prediction_results.csv')

    if not results_file_path or not os.path.exists(results_file_path):
        messages.error(request, "Results file not found or expired. Please run predictions again.")
        return redirect(reverse('core:researcher_upload'))

    try:
        # FileResponse handles streaming the file and is good for large files.
        response = FileResponse(open(results_file_path, 'rb'), as_attachment=True, filename=results_filename)

        # Clean up session and the temp file after starting the download.
        # Note: FileResponse does not guarantee the file is fully sent when it returns.
        # For more robust cleanup, consider background tasks or manage temp files carefully.
        # For this scope, immediate cleanup is acceptable.
        request.session.pop('results_file_path', None)
        request.session.pop('results_csv_filename', None)
        # os.remove(results_file_path) # This might be too soon if FileResponse is lazy.
        # A better approach for temp files is to use Django's signals (e.g. request_finished)
        # or a dedicated temporary file management strategy.
        # For now, we'll leave the file and it can be cleaned up if the user revisits upload page.

        return response
    except Exception as e:
        messages.error(request, f"Error downloading file: {e}")
        return redirect(reverse('core:researcher_results')) # Or back to researcher_overview
