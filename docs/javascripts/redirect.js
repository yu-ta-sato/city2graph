
(function() {
    if (window.location.pathname.endsWith('/index.html')) {
        var newPath = window.location.pathname.replace(/\/index\.html$/, '/');
        window.history.replaceState(null, '', newPath);
    }
})();
