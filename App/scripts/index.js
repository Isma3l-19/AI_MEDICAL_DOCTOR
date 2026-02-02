document.getElementById("form").onsubmit = async (e) => {
    e.preventDefault();

    const symptoms = document.getElementById("symptoms").value.trim();
    
    // Validate input
    if (!symptoms) {
        document.getElementById("result").innerHTML = 
            '<p class="text-red-600">Please enter symptoms</p>';
        return;
    }

    try {
        // Show loading state
        document.getElementById("result").innerHTML = 
            '<p class="text-gray-600">Analyzing...</p>';

        const res = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/x-www-form-urlencoded" },
            body: new URLSearchParams({ symptoms })
        });

        if (!res.ok) throw new Error("Network response failed");

        const data = await res.json();

        document.getElementById("result").innerHTML = `
            <p class="text-xl font-bold">Possible Condition: ${data.condition}</p>
            <p><b>Medicines:</b> ${data.medicines.join(", ")}</p>
            <p><b>Dosage:</b> ${data.dose}</p>
            <p><b>Timing:</b> ${data.timing}</p>
            <p class="text-sm text-red-600 mt-3">⚠️ ${data.disclaimer}</p>
        `;
    } catch (error) {
        document.getElementById("result").innerHTML = 
            `<p class="text-red-600">Error: ${error.message}</p>`;
    }
};