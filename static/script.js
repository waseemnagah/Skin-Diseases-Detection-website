// Get form and input elements
const form = document.getElementById('upload-form');
const fileInput = document.getElementById('file-input');

// Get result and error containers
const resultContainer = document.getElementById('result-container');
const resultClass = document.getElementById('result-class');
const confidenceScore = document.getElementById('confidence-score');
const errorContainer = document.getElementById('error-container');
const errorMessage = document.getElementById('error-message');

// Get image container and uploaded image
const imageContainer = document.getElementById('image-container');
const uploadedImage = document.getElementById('uploaded-image');

// Hide result and error containers
resultContainer.classList.add('hidden');
errorContainer.classList.add('hidden');

// Listen for form submit
form.addEventListener('submit', (event) => {
    event.preventDefault(); // Prevent form from submitting
    const file = fileInput.files[0];
    if (file) {
        // Create FormData object and append file to it
        const formData = new FormData();
        formData.append('file', file);

        // Send POST request to server
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                // Show result and hide error containers
                resultContainer.classList.remove('hidden');
                errorContainer.classList.add('hidden');

                // Set result and confidence score text
                resultClass.textContent = data.class;
                confidenceScore.textContent = `Confidence Score: ${data.confidence}`;

                // Set uploaded image source and show image container
                uploadedImage.src = URL.createObjectURL(file);
                imageContainer.classList.remove('hidden');
            })
            .catch(error => {
                // Show error and hide result containers
                errorContainer.classList.remove('hidden');
                resultContainer.classList.add('hidden');

                // Set error message text
                errorMessage.textContent = error.message;
            });
    }
});
