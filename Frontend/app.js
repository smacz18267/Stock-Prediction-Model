document.getElementById("predictionForm").addEventListener("submit", async function(e) {
    e.preventDefault();
    const ticker = document.getElementById("ticker").value;
    document.getElementById("result").innerText = "Processing prediction...";
    try {
        const response = await fetch("http://localhost:8000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ ticker: ticker })
        });
        const data = await response.json();
        document.getElementById("result").innerText = `Ticker: ${data.ticker}
Predicted Price: $${data.predicted_price.toFixed(2)}
Timestamp: ${data.timestamp}`;
    } catch (error) {
        document.getElementById("result").innerText = "Error retrieving prediction.";
        console.error(error);
    }
});
