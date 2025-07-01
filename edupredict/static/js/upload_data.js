document.addEventListener('DOMContentLoaded', function () {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-upload'); // From provided HTML
    // const form = fileInput ? fileInput.closest('form') : null; // Not strictly needed by this JS

    const dropzoneInitial = document.getElementById('dropzone-initial');
    const uploadProgressContainer = document.getElementById('upload-progress-container');
    const uploadFilename = document.getElementById('upload-filename');
    const uploadProgressBar = document.getElementById('upload-progress-bar');
    const uploadStatusMessage = document.getElementById('upload-status-message');
    const uploadSpinner = document.getElementById('upload-spinner');
    const uploadSuccessIcon = document.getElementById('upload-success-icon');

    const uploadErrorMessage = document.getElementById('upload-error-message');
    const errorText = document.getElementById('error-text');
    const uploadSuccessMessage = document.getElementById('upload-success-message');
    // const successText = document.getElementById('success-text'); // Element exists, not directly used by this JS logic for text

    const dataPreviewTable = document.querySelector('#data-preview-body')?.closest('div.overflow-x-auto'); // Get the table container
    const dataPreviewBody = document.getElementById('data-preview-body');
    const noDataMessage = document.getElementById('no-data-message');
    const continueButton = document.getElementById('continue-button');

    // Ensure all critical elements are present before proceeding
    if (!dropzone || !fileInput || !dropzoneInitial || !uploadProgressContainer || !uploadFilename ||
        !uploadProgressBar || !uploadStatusMessage || !uploadSpinner || !uploadSuccessIcon ||
        !uploadErrorMessage || !errorText || !uploadSuccessMessage ||
        !dataPreviewTable || !dataPreviewBody || !noDataMessage || !continueButton) {
        console.warn('One or more UI elements for file upload are missing. Full functionality may not be available.');
        // Optionally, disable the dropzone or show a general error if critical parts are missing
        if (dropzone) dropzone.style.pointerEvents = 'none'; // Prevent interaction if setup is incomplete
        return;
    }

    // Initial UI state from the provided HTML script
    if (dataPreviewTable) dataPreviewTable.style.display = 'none';
    if (noDataMessage) noDataMessage.style.display = 'block';
    if (continueButton) continueButton.disabled = true;


    function displayError(message) {
        errorText.textContent = message;
        uploadErrorMessage.classList.remove('hidden');
        uploadErrorMessage.classList.add('flex'); // Assuming flex is used for alignment

        dropzoneInitial.classList.remove('hidden');
        uploadProgressContainer.classList.add('hidden');
        if (dataPreviewTable) dataPreviewTable.style.display = 'none';
        noDataMessage.style.display = 'block';
        continueButton.disabled = true;
    }

    function displaySuccessUI(file) { // Renamed to avoid conflict if a server response is handled later
        uploadStatusMessage.textContent = 'Upload complete!'; // Or "File selected!" if not simulating upload yet
        uploadSuccessMessage.classList.remove('hidden');
        uploadSuccessMessage.classList.add('flex');

        if (dataPreviewTable) dataPreviewTable.style.display = 'block';
        noDataMessage.style.display = 'none';
        continueButton.disabled = false; // Enable continue button

        // Populate with dummy data for preview (as in provided HTML)
        const dummyData = [
            { id: 'S101', course: 'Calculus I', grade: 'A-', attendance: '92', hours: '11' },
            { id: 'S102', course: 'Intro to Programming', grade: 'B', attendance: '85', hours: '9' },
            { id: 'S103', course: 'World History', grade: 'B+', attendance: '90', hours: '7' },
            { id: 'S104', course: 'Physics Lab', grade: 'A', attendance: '98', hours: '14' },
            { id: 'S105', course: 'English Composition', grade: 'C+', attendance: '80', hours: '6' },
        ];
        dataPreviewBody.innerHTML = ''; // Clear previous preview
        dummyData.forEach(row => {
            const tr = document.createElement('tr');
            // Adapting to dark theme text from base.html (text-slate-300 by default for td)
            tr.className = 'hover:bg-slate-700 transition-colors';
            // Using text-slate-300 for data cells to match dark theme
            tr.innerHTML = `
              <td class="p-3 text-slate-300 whitespace-nowrap">${row.id}</td>
              <td class="p-3 text-slate-300 whitespace-nowrap">${row.course}</td>
              <td class="p-3 text-slate-300 whitespace-nowrap">${row.grade}</td>
              <td class="p-3 text-slate-300 whitespace-nowrap">${row.attendance}</td>
              <td class="p-3 text-slate-300 whitespace-nowrap">${row.hours}</td>
            `;
            dataPreviewBody.appendChild(tr);
        });
    }

    function handleFile(file) {
        if (!file) return;

        // Client-side validation (more robust than just filename includes)
        if (!file.type.startsWith('text/csv') && !file.name.toLowerCase().endsWith('.csv')) {
            displayError("Invalid file type. Only CSV files (.csv) are accepted.");
            fileInput.value = ''; // Reset file input to allow re-selection of same file if needed
            return;
        }
        const maxSize = 50 * 1024 * 1024; // 50MB
        if (file.size > maxSize) {
            displayError(`File is too large (${(file.size / (1024*1024)).toFixed(2)} MB). Maximum size is 50MB.`);
            fileInput.value = '';
            return;
        }

        uploadErrorMessage.classList.add('hidden');
        uploadSuccessMessage.classList.add('hidden');
        dropzoneInitial.classList.add('hidden');
        uploadProgressContainer.classList.remove('hidden');
        uploadSpinner.classList.remove('hidden');
        uploadSuccessIcon.classList.add('hidden');

        uploadProgressBar.style.width = '0%';
        uploadProgressBar.textContent = '0%';
        uploadFilename.textContent = file.name;
        uploadStatusMessage.textContent = 'Uploading...';
        continueButton.disabled = true;

        let progress = 0;
        const interval = setInterval(() => {
            progress += 10;
            uploadProgressBar.style.width = progress + '%';
            uploadProgressBar.textContent = progress + '%';
            if (progress >= 100) {
                clearInterval(interval);
                uploadStatusMessage.textContent = 'Processing...';
                uploadSpinner.classList.add('hidden');
                uploadSuccessIcon.classList.remove('hidden');

                setTimeout(() => { // Simulate processing delay
                    // These filename checks are from the provided HTML's simulation
                    if (file.name.toLowerCase().includes("error")) {
                        displayError("Invalid file format. Please upload a valid CSV file. Ensure all required columns (Student ID, Course, Grade, Attendance, Study Hours) are present.");
                        fileInput.value = '';
                    } else if (file.name.toLowerCase().includes("missing_column")) {
                        displayError("Missing required columns. Please ensure your CSV includes: Student ID, Course, Grade, Attendance, Study Hours.");
                        fileInput.value = '';
                    } else {
                        displaySuccessUI(file);
                    }
                }, 1000); // Simulate 1 sec processing
            }
        }, 200); // Simulate progress update interval
    }

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault(); // Necessary to allow drop
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            const file = e.dataTransfer.files[0];
            // Assign to the actual file input so the form submission includes it
            fileInput.files = e.dataTransfer.files;
            handleFile(file); // UI updates
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    // Allow clicking the dropzone (but not its interactive children like label) to trigger file input
    dropzone.addEventListener('click', (e) => {
        if (e.target.id === 'dropzone' || e.target.id === 'dropzone-initial' || e.target.parentElement.id === 'dropzone-initial') {
            // More precise: if not clicking on the label itself or elements within label
            if (!e.target.closest('label[for="file-upload"]')) {
                 fileInput.click();
            }
        }
    });
});
