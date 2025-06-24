document.addEventListener('DOMContentLoaded', function () {
    initializeFormValidation();
    initializeImagePreview();
    initializeFormSubmission();
});
function initializeFormValidation() {
    const forms = document.querySelectorAll('.needs-validation');
    Array.from(forms).forEach(form => {
        form.addEventListener('submit', event => {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        }, false);
    });
}
function initializeImagePreview() {
    const imageInput = document.getElementById('image');
    const imagePreview = document.getElementById('imagePreview');
    const previewImg = document.getElementById('previewImg');

    if (imageInput && imagePreview && previewImg) {
        imageInput.addEventListener('change', function (event) {
            const file = event.target.files[0];

            if (file) {
                const allowedTypes = ['image/png', 'image/jpg', 'image/jpeg', 'image/gif'];
                if (!allowedTypes.includes(file.type)) {
                    showError('Please select a valid image file (PNG, JPG, JPEG, or GIF).');
                    clearImagePreview();
                    return;
                }

                const maxSize = 10 * 1024 * 1024;
                if (file.size > maxSize) {
                    showError('File size is too large. Please select an image under 10MB.');
                    clearImagePreview();
                    return;
                }

                const reader = new FileReader();
                reader.onload = function (e) {
                    previewImg.src = e.target.result;
                    imagePreview.style.display = 'block';

                    imagePreview.style.opacity = '0';
                    setTimeout(() => {
                        imagePreview.style.transition = 'opacity 0.3s ease';
                        imagePreview.style.opacity = '1';
                    }, 10);
                };
                reader.readAsDataURL(file);

                                clearErrors();
            } else {
                clearImagePreview();
            }
        });
    }
}

function initializeFormSubmission() {
    const uploadForm = document.getElementById('uploadForm');
    const submitBtn = document.getElementById('submitBtn');
    const loadingSpinner = document.getElementById('loadingSpinner');

    if (uploadForm && submitBtn && loadingSpinner) {
        uploadForm.addEventListener('submit', function (event) {
            const imageInput = document.getElementById('image');

            if (!imageInput.files[0]) {
                event.preventDefault();
                showError('Please select an image file before submitting.');
                return;
            }

            showLoadingState(submitBtn, loadingSpinner);
        });
    }
}
function showLoadingState(submitBtn, loadingSpinner) {
        submitBtn.disabled = true;
    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';

        loadingSpinner.style.display = 'block';

        const imagePreview = document.getElementById('imagePreview');
    if (imagePreview) {
        imagePreview.style.display = 'none';
    }
}

function showError(message) {
        clearErrors();

        const errorAlert = document.createElement('div');
    errorAlert.className = 'alert alert-danger alert-dismissible fade show';
    errorAlert.setAttribute('role', 'alert');
    errorAlert.innerHTML = `
        <i class="fas fa-exclamation-triangle me-2"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;

        const form = document.getElementById('uploadForm');
    if (form) {
        form.parentNode.insertBefore(errorAlert, form);
    }

        setTimeout(() => {
        if (errorAlert.parentNode) {
            errorAlert.remove();
        }
    }, 5000);
}

function clearErrors() {
    const errorAlerts = document.querySelectorAll('.alert-danger');
    errorAlerts.forEach(alert => {
        if (alert.parentNode) {
            alert.remove();
        }
    });
}

function clearImagePreview() {
    const imagePreview = document.getElementById('imagePreview');
    const previewImg = document.getElementById('previewImg');

    if (imagePreview && previewImg) {
        imagePreview.style.display = 'none';
        previewImg.src = '';
    }
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';

    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function validateImageFile(file) {
    const allowedTypes = ['image/png', 'image/jpg', 'image/jpeg', 'image/gif'];
    const maxSize = 10 * 1024 * 1024;

    if (!allowedTypes.includes(file.type)) {
        return {
            valid: false,
            message: 'Please select a valid image file (PNG, JPG, JPEG, or GIF).'
        };
    }

    if (file.size > maxSize) {
        return {
            valid: false,
            message: `File size (${formatFileSize(file.size)}) is too large. Please select an image under 10MB.`
        };
    }

    return { valid: true };
}
const style = document.createElement('style');
style.textContent = `
    .drag-over {
        border: 2px dashed #4a90e2 !important;
        background-color: rgba(74, 144, 226, 0.1) !important;
        transform: scale(1.02);
        transition: all 0.3s ease;
    }
    
    .form-control:invalid {
        border-color: #dc3545;
        box-shadow: 0 0 0 0.2rem rgba(220, 53, 69, 0.25);
    }
    
    .form-control:valid {
        border-color: #28a745;
        box-shadow: 0 0 0 0.2rem rgba(40, 167, 69, 0.25);
    }
`;
document.head.appendChild(style);
