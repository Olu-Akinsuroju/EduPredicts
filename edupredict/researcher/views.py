# Researcher views will be defined here.
# Initial content, will be populated in a later step.
from django.shortcuts import render
from django.contrib.auth.decorators import login_required, user_passes_test
from django.conf import settings

# Helper for user_passes_test
def is_staff_user(user):
    return user.is_staff

@login_required(login_url=settings.LOGIN_URL)
@user_passes_test(is_staff_user, login_url=settings.LOGIN_URL)
def overview(request):
    """
    Renders the researcher overview page, which provides guidelines
    for data submission and a link to download a sample CSV file.
    Access is restricted to authenticated staff users.
    """
    return render(request, 'researcher/overview.html')
