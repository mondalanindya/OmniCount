document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.image-compare-container').forEach(container => {
        const slider = container.querySelector('.slider');
        const overlay = container.querySelector('.overlay');
        let isDown = false;

        slider.addEventListener('mousedown', (e) => {
            isDown = true;
            e.preventDefault();
            document.addEventListener('mousemove', mouseMoveHandler);
            document.addEventListener('mouseup', () => {
                isDown = false;
                document.removeEventListener('mousemove', mouseMoveHandler);
            }, { once: true });
        });

        const mouseMoveHandler = (e) => {
            if (!isDown) return;
            const rect = container.getBoundingClientRect();
            const x = e.pageX - rect.left;
            const walk = Math.max(0, Math.min(x, rect.width));
            slider.style.left = `${walk}px`;
            overlay.style.clipPath = `inset(0 ${rect.width - walk}px 0 0)`;
        };
    });
});
