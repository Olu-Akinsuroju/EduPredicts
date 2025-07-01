from django import forms

# Choices for ChoiceFields
STUDY_TIME_CHOICES = [
    ('', '---------'),
    (1, '<2 hours/week'),
    (2, '2-5 hours/week'),
    (3, '5-10 hours/week'),
    (4, '>10 hours/week'),
]

MOTIVATION_CHOICES = [
    ('', '---------'),
    (1, '1 (Lowest)'),
    (2, '2'),
    (3, '3 (Average)'),
    (4, '4'),
    (5, '5 (Highest)'),
]

PARENT_EDUCATION_CHOICES = [
    ('', '---------'),
    (0, 'None'),
    (1, 'Primary Education (Grade 1-4)'),
    (2, 'Lower Secondary Education (Grade 5-9)'),
    (3, 'Upper Secondary Education (High School Diploma or GED)'),
    (4, 'Vocational Training/Associate\'s Degree'),
    (5, 'Bachelor\'s Degree'),
    (6, 'Master\'s Degree'),
    (7, 'Doctorate (PhD) or Higher'),
]

# Tailwind CSS classes for form widgets to match base.html dark theme
TEXT_INPUT_CLASSES = 'mt-1 block w-full rounded-md border-slate-600 bg-slate-700 py-2 px-3 shadow-sm focus:border-sky-500 focus:outline-none focus:ring-sky-500 sm:text-sm text-slate-100 disabled:opacity-50'
SELECT_CLASSES = TEXT_INPUT_CLASSES
CHECKBOX_CLASSES = 'h-5 w-5 rounded border-slate-500 bg-slate-700 text-sky-600 focus:ring-sky-500 focus:ring-offset-slate-800 disabled:opacity-50'


class StudentInfoForm(forms.Form):
    previous_grade = forms.IntegerField(
        label="Previous Grade (0-100)",
        min_value=0,
        max_value=100,
        widget=forms.NumberInput(attrs={
            'class': TEXT_INPUT_CLASSES,
            'placeholder': 'e.g., 85',
        }),
        help_text="Enter your most recent overall grade or percentage."
    )
    absences = forms.IntegerField(
        label="Number of Absences",
        min_value=0,
        widget=forms.NumberInput(attrs={
            'class': TEXT_INPUT_CLASSES,
            'placeholder': 'e.g., 3',
        }),
        help_text="Number of school days missed in the current/last academic year."
    )
    study_time = forms.ChoiceField(
        label="Weekly Study Time",
        choices=STUDY_TIME_CHOICES,
        widget=forms.Select(attrs={
            'class': SELECT_CLASSES,
        }),
        help_text="Average hours spent studying per week."
    )
    age = forms.IntegerField(
        label="Age (Years)",
        min_value=10,
        max_value=30,
        widget=forms.NumberInput(attrs={
            'class': TEXT_INPUT_CLASSES,
            'placeholder': 'e.g., 16',
        }),
        help_text="Your current age in years."
    )
    internet_access = forms.BooleanField(
        label="Reliable Internet Access at Home",
        required=False,
        widget=forms.CheckboxInput(attrs={
            'class': CHECKBOX_CLASSES,
        }),
        help_text="Do you have consistent and reliable internet access at home?"
    )
    extra_courses = forms.BooleanField(
        label="Taking Extra Curricular Courses/Activities",
        required=False,
        widget=forms.CheckboxInput(attrs={
            'class': CHECKBOX_CLASSES,
        }),
        help_text="Are you enrolled in any courses or activities outside regular school hours?"
    )
    motivation_level = forms.ChoiceField( # Changed from 'motivation' to 'motivation_level'
        label="Motivation Level (1-5)",
        choices=MOTIVATION_CHOICES,
        widget=forms.Select(attrs={
            'class': SELECT_CLASSES,
        }),
        help_text="Rate your current motivation for studies on a scale of 1 to 5."
    )
    parent_education = forms.ChoiceField(
        label="Highest Parent Education Level",
        choices=PARENT_EDUCATION_CHOICES,
        widget=forms.Select(attrs={
            'class': SELECT_CLASSES,
        }),
        help_text="What is the highest level of education completed by any of your parents/guardians?"
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add aria-describedby for fields with help text for accessibility
        for field_name, field in self.fields.items():
            if field.help_text:
                # Construct an ID for the help text paragraph, assuming it follows a pattern in the template
                # Or, ensure the template manually adds id="{{ form.field_name.id_for_label }}_help_text"
                field.widget.attrs['aria-describedby'] = f'{field.auto_id}_help_text'


class ModelSelectionForm(forms.Form):
    MODEL_CHOICES = [
        ('lr', 'Logistic Regression'), # Value changed from 'logistic'
        ('dt', 'Decision Tree'),       # Value changed from 'tree'
        ('rf', 'Random Forest'),
    ]
    model_choice = forms.ChoiceField(  # Field name changed from 'model_name'
        label="Select Prediction Model", # This label won't be visible with manual template rendering
        choices=MODEL_CHOICES,
        widget=forms.RadioSelect, # Removed attrs as template will handle styling
        error_messages={'required': 'Please select a model to continue.'}
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Manual rendering in the template gives more control than trying to style RadioSelect here
        # for the complex card layout.
        # If we were using {{ form.model_choice.label_tag }} and {{ form.model_choice }},
        # then styling via widgets would be more relevant.
        pass # Keep __init__ simple as template handles rendering

# Make MODEL_CHOICES globally available in this module for import by views.py
MODEL_CHOICES = ModelSelectionForm.MODEL_CHOICES

class FileUploadForm(forms.Form):
    csv_file = forms.FileField(
        label="Upload Your CSV File",
        required=True,
        widget=forms.FileInput(attrs={
            'id': 'file-upload',
            'class': 'block w-full text-sm text-slate-400 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-sky-500 file:text-sky-50 hover:file:bg-sky-600 focus:outline-none focus:ring-2 focus:ring-sky-500 focus:ring-offset-2 focus:ring-offset-slate-800 disabled:opacity-50'
            # Simplified classes, assuming direct rendering or that template will style the label separately
        })
    )
    # Add model_choice field for researcher CSV upload
    model_choice = forms.ChoiceField(
        label="Select Prediction Model",
        choices=MODEL_CHOICES, # Reuse from ModelSelectionForm
        required=True,
        widget=forms.Select(attrs={ # Using Select for a more compact UI in this context
            'class': SELECT_CLASSES, # Reuse existing styling
        }),
        initial='lr', # Default to Logistic Regression
        help_text="Choose the model to use for predictions on the uploaded CSV data."
    )

    def clean_csv_file(self):
        uploaded_file = self.cleaned_data.get('csv_file')
        if uploaded_file:
            if not uploaded_file.name.endswith('.csv'):
                raise forms.ValidationError("Only CSV files are allowed.")
        # It's important to return the cleaned data, whether it's modified or not.
        # If the file is valid or not present (and not required), return it.
        # If it's None and required, FileField itself will raise a validation error.
        return uploaded_file
