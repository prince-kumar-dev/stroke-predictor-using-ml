document.getElementById('strokeForm').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent the default form submission

    const resultDiv = document.getElementById('result');
    const errorDiv = document.getElementById('error');
    const predictionTextEl = document.getElementById('predictionText'); // Renamed variable for clarity
    const probabilityTextEl = document.getElementById('probabilityText'); // Renamed variable for clarity
    const errorTextEl = document.getElementById('errorText'); // Renamed variable for clarity

    // Clear previous results and errors
    resultDiv.style.display = 'none';
    errorDiv.style.display = 'none';
    predictionTextEl.textContent = '';
    probabilityTextEl.textContent = '';
    errorTextEl.textContent = '';
    // **Remove previous risk classes**
    resultDiv.classList.remove('low-risk', 'medium-risk', 'high-risk');

    // Collect form data (Ensure this part is correct from previous steps)
    const formData = new FormData(this);
    const data = {};
    formData.forEach((value, key) => {
        const radioButtons = document.querySelectorAll(`input[name="${key}"][type="radio"]`);
        if (radioButtons.length > 0) {
             const checkedRadio = document.querySelector(`input[name="${key}"]:checked`);
             if (checkedRadio) { data[key] = checkedRadio.value; }
        } else {
            if (key === 'bmi' && value === '') { data[key] = null; }
            else { data[key] = value; }
        }
    });
    // Ensure binary features are included even if default 'No' (value=0) wasn't explicitly clicked sometimes
    ['hypertension', 'heart_disease', 'ever_married', 'Residence_type'].forEach(key => {
        if (!data.hasOwnProperty(key)) {
             const checkedRadio = document.querySelector(`input[name="${key}"]:checked`);
             if (checkedRadio) { data[key] = checkedRadio.value;}
             // Consider adding default '0' or 'No' if nothing is checked and required? Handled by 'required' for now.
        }
    });


    console.log('Sending data:', data);

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => {
                throw new Error(err.error || `HTTP error! Status: ${response.status}`);
            }).catch(() => {
                 throw new Error(`HTTP error! Status: ${response.status}`);
            });
        }
        return response.json();
    })
    .then(predictionResult => {
        console.log('Received prediction:', predictionResult);

        // Update text content
        predictionTextEl.textContent = predictionResult.prediction_text; // Use text from backend
        probabilityTextEl.textContent = `Probability of Stroke: ${predictionResult.probability_stroke}%`;

        // **Add the correct class based on the risk level from backend**
        if (predictionResult.risk_level) {
            resultDiv.classList.add(predictionResult.risk_level + '-risk'); // e.g., adds 'low-risk', 'medium-risk', or 'high-risk'
        }

        // Display the result box
        resultDiv.style.display = 'block';
        errorDiv.style.display = 'none';
    })
    .catch(error => {
        console.error('Error during prediction fetch:', error);
        errorTextEl.textContent = `Prediction failed: ${error.message}`;
        errorDiv.style.display = 'block';
        resultDiv.style.display = 'none';
    });
});