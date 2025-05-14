document.addEventListener("DOMContentLoaded", function () {
    const formElement = document.getElementById('diabetes-form');

    if (!formElement) {
        console.error("Form with ID 'diabetes-form' not found.");
        return;
    }

    formElement.addEventListener('submit', async function (e) {
        e.preventDefault();

        // Get form values
        const formData = {
            pregnancies: document.getElementById('pregnancies').value,
            glucose: document.getElementById('glucose').value,
            bloodPressure: document.getElementById('bloodpressure').value,
            skinthickness: document.getElementById('skinthickness').value,
            insulin: document.getElementById('insulin').value,
            bmi: document.getElementById('bmi').value,
            diabetesPedigreeFunction: document.getElementById('dpf').value,
            age: document.getElementById('age').value
        };

        console.log('Sending data to server:', formData);

        const maxRetries = 3;
        let retryCount = 0;

        async function tryPrediction() {
            try {
                const response = await fetch('http://127.0.0.1:5000/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Accept': 'application/json'
                    },
                    mode: 'cors',
                    credentials: 'omit',
                    body: JSON.stringify(formData)
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const result = await response.json();
                console.log('Server response:', result);

                // Display result
                const resultDiv = document.getElementById('prediction-result');
                const resultSpan = document.getElementById('result');

                resultDiv.style.display = 'block';
                if (result.prediction === 1) {
                    resultDiv.className = 'positive';
                    resultSpan.textContent = `High Risk of Diabetes (${result.risk_percentage}% probability)`;
                } else {
                    resultDiv.className = 'negative';
                    resultSpan.textContent = `Low Risk of Diabetes (${100 - result.risk_percentage}% probability)`;
                }
            } catch (error) {
                console.error('Error:', error);
                if (retryCount < maxRetries) {
                    retryCount++;
                    console.log(`Retrying... Attempt ${retryCount} of ${maxRetries}`);
                    await new Promise(resolve => setTimeout(resolve, 1000 * retryCount));
                    return tryPrediction();
                }
                alert("Error: Failed to fetch. Please make sure the server is running at http://127.0.0.1:5000");
            }
        }

        tryPrediction();
    });
});
