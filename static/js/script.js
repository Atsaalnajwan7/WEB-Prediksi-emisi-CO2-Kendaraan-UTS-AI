// Custom JavaScript untuk Aplikasi Prediksi CO2

// Loading indicator saat form submit
document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('predictionForm');
    const loadingDiv = document.getElementById('loading');
    const submitBtn = document.querySelector('button[type="submit"]');

    if (form) {
        form.addEventListener('submit', function () {
            if (loadingDiv) {
                loadingDiv.style.display = 'block';
            }
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Memproses...';
            }
        });
    }
});

// Smooth scroll untuk anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});