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
        ('logistic', 'Logistic Regression'),
        ('tree', 'Decision Tree'),
        ('rf', 'Random Forest'),
    ]
    model_name = forms.ChoiceField(
        label="Select Prediction Model",
        choices=MODEL_CHOICES,
        widget=forms.RadioSelect(attrs={'class': 'mr-2'})
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Tailwind styling for radio buttons is tricky with default widget rendering.
        # For better control, one might render radio inputs manually in the template.
        # This basic class attribute is a starting point.
        for field_name, field in self.fields.items():
             # General styling for the field container if needed
            field.widget.attrs.update({'class': 'space-y-2'})


class FileUploadForm(forms.Form):
    file = forms.FileField(label="Upload CSV File")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field_name, field in self.fields.items():
            field.widget.attrs.update({'class': 'mt-1 block w-full text-sm text-slate-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-violet-50 file:text-violet-700 hover:file:bg-violet-100'})

    def clean_file(self):
        file = self.cleaned_data.get('file')
        if file:
            if not file.name.endswith('.csv'):
                raise forms.ValidationError("Only CSV files are allowed.")
        return file
