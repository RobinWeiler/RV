alert('If you see this alert, then your custom JavaScript script has run!');

// Function to handle changes in the data-dash-is-loading attribute


// Select the specific div with the data-dash-is-loading attribute by id
var loadingElement = document.getElementById('EEG-graph');

// Get the target div by id
//   const targetDiv = document.getElementById('loader');

// Create a MutationObserver with the callback function
var observer = new MutationObserver(function(mutations) {
    mutations.forEach(function(mutation) {
        alert('hello');
        if (mutation.type === 'attributes' && mutation.attributeName === 'data-dash-is-loading') {
            // Get the new value of the attribute
            const newAttributeValue = mutation.target.getAttribute('data-dash-is-loading');
    
            // Get the target div by id
            const targetDiv = document.getElementById('loader');
    
            // Update the display property based on the attribute value
            if (newAttributeValue === 'true') {
                targetDiv.style.display = 'block';
                alert('Here');
            } else {
                targetDiv.style.display = 'none';
                alert('There');
            }
        }
    });
});

// Observe changes to the attributes of the specific loading element
observer.observe(loadingElement, { attributes: true });

// Initial check for the current value of data-dash-is-loading
// handleLoadingAttributeChange([{ target: loadingElement }], observer);