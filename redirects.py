(function() {
    // Function to intercept window.location changes
    let originalLocation = window.location;
    
    Object.defineProperty(window, "location", {
        get: function() {
            return originalLocation;
        },
        set: function(value) {
            console.log("Redirect attempt to:", value);
            // Optionally, you can show an alert or log the redirect attempt
            // alert("Attempt to redirect to: " + value);
            // If you want to prevent the redirect, simply do nothing here
        }
    });

    // Intercept window.location.href assignments
    let originalHREF = Object.getOwnPropertyDescriptor(window.location, 'href').set;
    
    Object.defineProperty(window.location, 'href', {
        get: function() {
            return originalLocation.href;
        },
        set: function(value) {
            console.log("Redirect attempt to:", value);
            // Optionally, you can show an alert or log the redirect attempt
            // alert("Attempt to redirect to: " + value);
            // If you want to prevent the redirect, simply do nothing here
        }
    });

    // Intercept document.location changes
    let originalDocumentLocation = window.document.location;

    Object.defineProperty(window.document, "location", {
        get: function() {
            return originalDocumentLocation;
        },
        set: function(value) {
            console.log("Redirect attempt to:", value);
            // Optionally, you can show an alert or log the redirect attempt
            // alert("Attempt to redirect to: " + value);
            // If you want to prevent the redirect, simply do nothing here
        }
    });

    // Intercept document.location.href assignments
    let originalDocumentHREF = Object.getOwnPropertyDescriptor(window.document.location, 'href').set;
    
    Object.defineProperty(window.document.location, 'href', {
        get: function() {
            return originalDocumentLocation.href;
        },
        set: function(value) {
            console.log("Redirect attempt to:", value);
            // Optionally, you can show an alert or log the redirect attempt
            // alert("Attempt to redirect to: " + value);
            // If you want to prevent the redirect, simply do nothing here
        }
    });

    // Intercept window.open calls
    let originalOpen = window.open;
    
    window.open = function(url, target, features, replace) {
        console.log("Window open attempt to:", url);
        // Optionally, you can show an alert or log the window open attempt
        // alert("Attempt to open new window with URL: " + url);
        // If you want to prevent the window from opening, simply do nothing here
    };

    // Intercept meta-refresh tags
    let observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.type === 'childList') {
                mutation.addedNodes.forEach(function(node) {
                    if (node.tagName && node.tagName.toLowerCase() === 'meta' && node.getAttribute('http-equiv').toLowerCase() === 'refresh') {
                        console.log("Meta-refresh tag detected and removed:", node);
                        // Optionally, you can show an alert or log the meta-refresh attempt
                        // alert("Meta-refresh tag detected and removed");
                        node.parentNode.removeChild(node);
                    }
                });
            }
        });
    });

    observer.observe(document.documentElement, { childList: true, subtree: true });

    // Intercept setTimeout redirects
    let originalSetTimeout = window.setTimeout;
    
    window.setTimeout = function(func, delay) {
        if (typeof func === 'function') {
            let newFunc = function() {
                try {
                    func();
                } catch (e) {
                    console.log("Function in setTimeout failed:", e);
                }
            };
            return originalSetTimeout(newFunc, delay);
        } else {
            return originalSetTimeout(func, delay);
        }
    };

    // Intercept setInterval redirects
    let originalSetInterval = window.setInterval;
    
    window.setInterval = function(func, interval) {
        if (typeof func === 'function') {
            let newFunc = function() {
                try {
                    func();
                } catch (e) {
                    console.log("Function in setInterval failed:", e);
                }
            };
            return originalSetInterval(newFunc, interval);
        } else {
            return originalSetInterval(func, interval);
        }
    };

    // Intercept location assignments
    let originalAssign = window.location.assign;
    
    window.location.assign = function(url) {
        console.log("Location assign attempt to:", url);
        // Optionally, you can show an alert or log the location assign attempt
        // alert("Attempt to assign new URL: " + url);
        // If you want to prevent the redirect, simply do nothing here
    };

    // Intercept location.replace calls
    let originalReplace = window.location.replace;
    
    window.location.replace = function(url) {
        console.log("Location replace attempt to:", url);
        // Optionally, you can show an alert or log the location replace attempt
        // alert("Attempt to replace URL: " + url);
        // If you want to prevent the redirect, simply do nothing here
    };

    console.log("Redirect prevention script loaded.");
})();
