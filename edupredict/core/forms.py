from django import forms

class StudentInfoForm(forms.Form):
    previous_grade = forms.FloatField(label="Previous Grade (0-100)", min_value=0, max_value=100)
    absences = forms.IntegerField(label="Number of Absences", min_value=0)
    study_time = forms.FloatField(label="Weekly Study Time (hours)", min_value=0)
    age = forms.IntegerField(label="Age", min_value=5, max_value=100)
    internet_access = forms.BooleanField(label="Has Internet Access at Home", required=False)
    extra_courses = forms.BooleanField(label="Takes Extra Courses", required=False)
    motivation = forms.IntegerField(label="Motivation Level (1-5)", min_value=1, max_value=5)
    parent_education = forms.IntegerField(label="Parent's Education Level (1-5, 1=None, 5=Postgraduate)", min_value=1, max_value=5)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field_name, field in self.fields.items():
            field.widget.attrs.update({'class': 'mt-1 block w-full px-3 py-2 bg-white border border-slate-300 rounded-md text-sm shadow-sm placeholder-slate-400 focus:outline-none focus:border-sky-500 focus:ring-1 focus:ring-sky-500'})

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
