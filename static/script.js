document.addEventListener('DOMContentLoaded', function() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const imagePreviews = document.getElementById('image-previews');
    const clearButton = document.getElementById('clear-button');
    const predictButton = document.getElementById('predict-button');
    const spinner = predictButton.querySelector('.spinner');
    const errorMessage = document.getElementById('error-message');
    const uploadForm = document.getElementById('upload-form');

    // Handle file selection via click
    dropZone.addEventListener('click', () => fileInput.click());

    // Handle drag and drop events
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        
        if (e.dataTransfer.files.length > 0) {
            fileInput.files = e.dataTransfer.files;
            handleFileSelect();
        }
    });

    // Handle file selection via input
    fileInput.addEventListener('change', handleFileSelect);

    // Clear all selected files
    clearButton.addEventListener('click', () => {
        clearFiles();
    });

    // Form submission
    uploadForm.addEventListener('submit', (e) => {
        if (fileInput.files.length === 0) {
            e.preventDefault();
            showError('Please select at least one image to analyze.');
            return;
        }

        // Show loading state
        predictButton.querySelector('span').textContent = 'Processing...';
        spinner.classList.remove('hidden');
        predictButton.disabled = true;
    });

    // Handle file selection
    function handleFileSelect() {
        const files = fileInput.files;
        
        if (files.length === 0) {
            previewContainer.classList.add('hidden');
            return;
        }

        // Clear previous error messages
        hideError();
        
        // Validate file types
        const invalidFiles = Array.from(files).filter(file => {
            const fileType = file.type.toLowerCase();
            return !['image/jpeg', 'image/jpg', 'image/png'].includes(fileType);
        });

        if (invalidFiles.length > 0) {
            showError('Only JPG, JPEG, and PNG files are allowed.');
            clearFiles();
            return;
        }

        // Clear previous previews
        imagePreviews.innerHTML = '';
        
        // Create previews for each file
        Array.from(files).forEach((file, index) => {
            const reader = new FileReader();
            
            reader.onload = function(e) {
                const preview = document.createElement('div');
                preview.className = 'image-preview';
                preview.innerHTML = `
                    <img src="${e.target.result}" alt="${file.name}">
                    <button type="button" class="remove-btn" data-index="${index}">×</button>
                `;
                imagePreviews.appendChild(preview);
                
                // Add event listener to remove button
                preview.querySelector('.remove-btn').addEventListener('click', function() {
                    removeFile(this.getAttribute('data-index'));
                });
            };
            
            reader.readAsDataURL(file);
        });
        
        previewContainer.classList.remove('hidden');
    }

    // Remove a specific file
    function removeFile(index) {
        const dt = new DataTransfer();
        const files = fileInput.files;
        
        for (let i = 0; i < files.length; i++) {
            if (i != index) {
                dt.items.add(files[i]);
            }
        }
        
        fileInput.files = dt.files;
        handleFileSelect();
        
        if (fileInput.files.length === 0) {
            previewContainer.classList.add('hidden');
        }
    }

    // Clear all files
    function clearFiles() {
        fileInput.value = '';
        imagePreviews.innerHTML = '';
        previewContainer.classList.add('hidden');
    }

    // Show error message
    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.classList.remove('hidden');
    }

    // Hide error message
    function hideError() {
        errorMessage.classList.add('hidden');
    }
});