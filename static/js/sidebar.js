function toggleSidebar(isExpanded) {
    const sidebar = document.querySelector('[data-testid="stSidebar"]');
    const content = document.querySelector('.content');
    const toggleBtn = document.querySelector('.toggle-btn');
    if (sidebar && content && toggleBtn) {
        if (isExpanded === undefined) {
            isExpanded = !sidebar.classList.contains('expanded');
        }
        if (isExpanded) {
            sidebar.classList.add('expanded');
            content.style.marginLeft = '250px';
            toggleBtn.textContent = '←'; // Change icon to indicate collapse
        } else {
            sidebar.classList.remove('expanded');
            content.style.marginLeft = '60px';
            toggleBtn.textContent = '→'; // Change icon to indicate expand
        }
    }
}

// Add click event listener to toggle button
document.addEventListener('DOMContentLoaded', function() {
    const toggleBtn = document.querySelector('.toggle-btn');
    if (toggleBtn) {
        toggleBtn.addEventListener('click', function() {
            toggleSidebar();
        });
    }
});