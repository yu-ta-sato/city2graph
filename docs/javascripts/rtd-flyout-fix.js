document.addEventListener("DOMContentLoaded", function() {
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            mutation.addedNodes.forEach(function(node) {
                if (node.tagName && node.tagName.toLowerCase() === 'readthedocs-flyout') {
                    injectStyles(node);
                }
            });
        });
    });

    observer.observe(document.body, { childList: true, subtree: true });

    // Check if already present
    const existing = document.querySelector('readthedocs-flyout');
    if (existing) injectStyles(existing);

    function injectStyles(flyout) {
        if (!flyout.shadowRoot) return;

        const style = document.createElement('style');
        style.textContent = `
            .logo {
                width: 2.5rem !important;
                height: auto !important;
            }
            .rst-versions .rst-other-versions, .versions {
                max-height: 300px !important;
                overflow-y: auto !important;
            }
        `;
        flyout.shadowRoot.appendChild(style);
    }
});
