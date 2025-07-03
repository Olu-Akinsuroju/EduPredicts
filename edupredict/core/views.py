from django.shortcuts import render, redirect
from django.urls import reverse
from django.http import HttpResponse, FileResponse
from django.contrib.auth.decorators import login_required
from .forms import (
    StudentInfoForm, ModelSelectionForm, FileUploadForm,
    STUDY_TIME_CHOICES, MOTIVATION_CHOICES, PARENT_EDUCATION_CHOICES, MODEL_CHOICES # Import choices
)
from .ml_utils import predict_student, batch_predict, get_sample_csv_data # Updated functions
import pandas as pd
import io
import logging # Import logging
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
    return render(request, 'student/form.html', {'form': form})


def student_select_model(request): # Renamed from student_select_model_view
    student_data = request.session.get('student_data')
    if not student_data: # Ensure student data exists from previous step
        messages.error(request, "Please submit your information first to select a model.")
        return redirect(reverse('core:student_info'))

    if request.method == 'POST':
        form = ModelSelectionForm(request.POST) # Uses the updated ModelSelectionForm
        if form.is_valid():
            selected_model_key = form.cleaned_data['model_choice']
            # Get the display name for the message
            model_display_name = dict(form.fields['model_choice'].choices).get(selected_model_key, "Selected Model")

            request.session['model_name'] = selected_model_key # Store key 'lr', 'dt', 'rf'
            messages.success(request, f"Model '{model_display_name}' selected. Proceeding to prediction.")
            return redirect(reverse('core:student_predict')) # Redirect to the actual prediction view
        else:
            # Form is invalid, re-render the page with errors
            messages.error(request, "Please correct the errors below and select a model.")
    else:
        # GET request, display an empty form
        form = ModelSelectionForm()

    # Render the new model selection page
    return render(request, 'student/model_selection.html', {'form': form})


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

    # Call the updated predict_student function from ml_utils
    # student_data is form.cleaned_data (a dict), model_name is 'lr', 'dt', or 'rf'
    prediction_output = predict_student(student_data, model_name)

    logger.debug(f"Prediction output for student: {prediction_output}")

    if prediction_output.get("error"):
        messages.error(request, f"Prediction failed: {prediction_output['error']}")
        # Redirect to model selection or form, allowing user to retry or change model
        return redirect(reverse('core:student_select_model'))

    # Successfully got prediction
    passed_bool = prediction_output.get('passed', False)
    probability_float = prediction_output.get('probability_score', 0.0)

    # Determine model display name for the results page
    model_display_name = dict(MODEL_CHOICES).get(model_name, "Selected Model")


    outcome_str = "Passed" if passed_bool else "Failed"
    probability_percent = round(probability_float * 100, 1)

    input_summary_dict = _prepare_input_summary(student_data) # Uses original student_data from session

    context = {
        'outcome': outcome_str,
        'probability': probability_percent,
        'input_summary': input_summary_dict,
        'model_name': model_display_name, # Display user-friendly name
        # student_data is implicitly used by _prepare_input_summary,
        # but not directly needed by template if input_summary is comprehensive.
    }
    return render(request, 'core/student_result.html', context)

def _prepare_input_summary(student_data):
    """
    Transforms raw student data from session into a user-friendly summary dictionary.
    """
    if not student_data:
        return {}

    # Create display maps from choices
    study_time_map = {k: v for k, v in STUDY_TIME_CHOICES if k}
    motivation_map = {k: v for k, v in MOTIVATION_CHOICES if k}
    parent_education_map = {k: v for k, v in PARENT_EDUCATION_CHOICES if k}

    summary = {
        "Previous Grade": student_data.get('previous_grade', 'N/A'),
        "Absences": student_data.get('absences', 'N/A'),
        "Weekly Study Time": study_time_map.get(student_data.get('study_time'), 'N/A'),
        "Age": student_data.get('age', 'N/A'),
        "Internet Access at Home": "Yes" if student_data.get('internet_access') else "No",
        "Taking Extra Courses": "Yes" if student_data.get('extra_courses') else "No",
        "Motivation Level": motivation_map.get(student_data.get('motivation_level'), 'N/A'),
        "Highest Parent Education": parent_education_map.get(student_data.get('parent_education'), 'N/A'),
    }
    return summary

@login_required(login_url=settings.LOGIN_URL)
def student_upload_data(request):
    if request.method == 'POST':
        form = FileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = form.cleaned_data.get('csv_file') # Get from cleaned_data after validation
            if uploaded_file: # Should always be true if form is_valid and field is required
                messages.success(request, f"File '{uploaded_file.name}' received.") # Updated message
            else:
                # This case should ideally not be reached if FileField is required and form.is_valid() passed.
                # However, as a fallback or if field was not required:
                messages.error(request, "File processing error: No file found after validation.")
            return redirect('core:student_upload_data')
        else:
            # Form is not valid. Errors are in form.errors.
            # The template should display form.csv_file.errors.
            # Adding a general message.
            if 'csv_file' in form.errors and any('This field is required.' in e for e in form.errors['csv_file']):
                 messages.error(request, "Upload failed: No file was submitted. Please choose a CSV file.")
            elif 'csv_file' in form.errors and any('The submitted file is empty.' in e for e in form.errors['csv_file']):
                 messages.error(request, "Upload failed: The submitted CSV file is empty.")
            else:
                 messages.error(request, "Upload failed. Please check errors or try a different CSV file.")
    else:
        form = FileUploadForm()

    return render(request, 'student/upload_data.html', {'form': form})


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
        form = FileUploadForm(request.POST, request.FILES) # This form now includes model_choice
        if form.is_valid():
            uploaded_file = form.cleaned_data['csv_file'] # Get from cleaned_data
            selected_model_key = form.cleaned_data['model_choice'] # Get selected model
            selected_model_display_name = dict(MODEL_CHOICES).get(selected_model_key, "Selected Model")

            logger.info(f"File uploaded: {uploaded_file.name}, Model selected for preview: {selected_model_key} ({selected_model_display_name})")

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

                # Store selected model key in session to be potentially used by researcher_results_view
                # if we decide to not rely on the hidden field in the template.
                # For now, template uses hidden field, but this is good for robustness.
                request.session['selected_model_key_researcher'] = selected_model_key


                return render(request, 'core/researcher_upload.html', {
                    'form': form, # Pass the form again, it might have errors if only one field was ok
                    'preview_html': preview_html,
                    'file_uploaded': True,
                    'selected_model_key': selected_model_key, # For the hidden input in the second form
                    'selected_model_display_name': selected_model_display_name # For display
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

        # Researcher model selection:
        # For now, let's allow the researcher to select a model via a POST parameter.
        # If not provided, default to 'lr'.
        # This assumes the form on researcher_upload.html will be updated to include model selection.
        # A more robust implementation might involve a separate ModelSelectionForm for researchers.
        selected_model_key = request.POST.get('model_choice', 'lr') # Default to 'lr'
        model_display_name = dict(MODEL_CHOICES).get(selected_model_key, "Selected Model")

        logger.info(f"Researcher selected model: {selected_model_key} ({model_display_name}) for batch prediction.")

        try:
            # Call the updated batch_predict function from ml_utils
            results_df, error_message = batch_predict(file_path, selected_model_key)

            if error_message:
                messages.error(request, f"Batch prediction failed: {error_message}")
                # Clean up uploaded file in case of error during prediction
                if os.path.exists(file_path):
                    os.remove(file_path)
                request.session.pop('uploaded_file_path', None)
                return redirect(reverse('core:researcher_upload'))

            if results_df.empty:
                messages.warning(request, "Batch prediction resulted in empty data. Please check the input file.")
                # Clean up uploaded file
                if os.path.exists(file_path):
                    os.remove(file_path)
                request.session.pop('uploaded_file_path', None)
                return redirect(reverse('core:researcher_upload'))

            # Prepare results for display (e.g., top 50 rows)
            results_preview_html = results_df.head(50).to_html(
                classes=['min-w-full', 'divide-y', 'divide-gray-200', 'border', 'border-gray-300', 'table-auto'],
                border=0, justify='left', index=False
            )

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
