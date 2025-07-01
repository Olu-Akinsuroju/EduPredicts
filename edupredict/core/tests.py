from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.conf import settings
from django.contrib.messages import get_messages

from .forms import FileUploadForm

class StudentUploadDataViewTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.student_user = User.objects.create_user(username='student', password='password123')
        self.staff_user = User.objects.create_user(username='staff', password='password123', is_staff=True)

        self.upload_url = reverse('core:student_upload_data')

        self._original_login_url = getattr(settings, 'LOGIN_URL', None)
        settings.LOGIN_URL = '/admin/login/'

    def tearDown(self):
        if self._original_login_url is not None:
            settings.LOGIN_URL = self._original_login_url
        else:
            if hasattr(settings, 'LOGIN_URL'):
                del settings.LOGIN_URL

    def test_student_upload_data_view_get_unauthenticated(self):
        """Test GET request by unauthenticated user redirects to LOGIN_URL."""
        response = self.client.get(self.upload_url)
        expected_redirect_url = f"{settings.LOGIN_URL}?next={self.upload_url}"
        self.assertRedirects(response, expected_redirect_url, status_code=302, target_status_code=200)

    def test_student_upload_data_view_get_authenticated_student(self):
        """Test GET request by authenticated student user."""
        self.client.login(username='student', password='password123')
        response = self.client.get(self.upload_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'student/upload_data.html')
        self.assertIsInstance(response.context['form'], FileUploadForm)

    def test_student_upload_data_view_post_no_file(self):
        """Test POST request with no file submitted by an authenticated student."""
        self.client.login(username='student', password='password123')
        response = self.client.post(self.upload_url, {})
        self.assertEqual(response.status_code, 200)
        self.assertFormError(response, 'form', 'csv_file', 'This field is required.')

        messages_list = list(get_messages(response.wsgi_request))
        self.assertTrue(any(message.level_tag == 'error' for message in messages_list))
        self.assertTrue(any("Upload failed" in message.message for message in messages_list)) # Corrected string

    def test_student_upload_data_view_post_invalid_file_type(self):
        """Test POST request with a non-CSV file by an authenticated student."""
        self.client.login(username='student', password='password123')
        txt_file = SimpleUploadedFile("test_file.txt", b"This is some text content.", content_type="text/plain")
        response = self.client.post(self.upload_url, {'csv_file': txt_file})

        self.assertEqual(response.status_code, 200)
        self.assertFormError(response, 'form', 'csv_file', 'Only CSV files are allowed.')

        messages_list = list(get_messages(response.wsgi_request))
        self.assertTrue(any(message.level_tag == 'error' for message in messages_list))
        self.assertTrue(any("Upload failed" in message.message for message in messages_list)) # Corrected string

    def test_student_upload_data_view_post_valid_csv_file(self):
        """Test POST request with a valid CSV file by an authenticated student."""
        self.client.login(username='student', password='password123')
        csv_content_str = (
            "Previous_Grade,Absences,Study_Time_per_Week,Age,passed,Gender_Male,Extra_Courses_Yes,Internet_Access_Yes,"
            "Motivation_Level_High,Motivation_Level_Low,Motivation_Level_Medium,"
            "Parent_Education_Level_Primary,Parent_Education_Level_Secondary,Parent_Education_Level_Tertiary,G3\n"
            "81.9,14,14.9,17,1,1,1,1,0,1,0,0,0,1,68.72700594236812\n"
            "72.2,8,12.7,16,1,0,1,1,0,0,1,0,0,1,97.53571532049581"
        )
        csv_content_bytes = csv_content_str.encode('utf-8')
        csv_file = SimpleUploadedFile("test_valid_data.csv", csv_content_bytes, content_type="text/csv")

        initial_response = self.client.post(self.upload_url, {'csv_file': csv_file})

        # Debugging print, can be removed once test passes
        if initial_response.status_code != 302:
            if 'form' in initial_response.context and initial_response.context['form'].errors:
                print(f"DEBUG test_post_valid_csv_file: Form errors: {initial_response.context['form'].errors}")
            else:
                print(f"DEBUG test_post_valid_csv_file: No form errors in context, or form is valid but still got {initial_response.status_code}. Context: {initial_response.context.keys() if 'form' in initial_response.context else 'form not in context'}")

        self.assertEqual(initial_response.status_code, 302, "Initial POST did not redirect. Check printed form errors if any.")
        self.assertEqual(initial_response.url, self.upload_url, f"Redirect URL is incorrect.")

        response_followed = self.client.get(initial_response.url)
        self.assertEqual(response_followed.status_code, 200, "Following redirect failed.")

        actual_messages = list(response_followed.context.get('messages', []))
        self.assertTrue(any(message.level_tag == 'success' for message in actual_messages),
                        f"No success messages found in context. Messages: {[str(m) for m in actual_messages]}")

        expected_message_content = f"File '{csv_file.name}' received."

        self.assertTrue(any(expected_message_content in message.message for message in actual_messages),
                        f"Success message ('{expected_message_content}') not found in context messages. Actual messages: {[str(m) for m in actual_messages]}")

        # Check if the key part of the message is rendered in the HTML response
        self.assertContains(response_followed, expected_message_content)


    def test_student_upload_data_view_get_authenticated_staff_as_student(self):
        """Test GET request by authenticated staff user (should behave like a student for this view)."""
        self.client.login(username='staff', password='password123')
        response = self.client.get(self.upload_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'student/upload_data.html')
        self.assertIsInstance(response.context['form'], FileUploadForm)
