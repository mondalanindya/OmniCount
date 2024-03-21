document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.image-compare-container').forEach(container => {
        let slider = container.querySelector('.slider');
        let overlay = container.querySelector('.overlay');
        let isPressed = false;

        const moveSlider = (e) => {
            if (!isPressed) return;
            // Calculate the new position of the slider.
            let rect = container.getBoundingClientRect();
            let newPos = e.clientX - rect.left;
            let width = rect.right - rect.left;

            // Limit the slider position to container bounds.
            newPos = Math.max(0, Math.min(newPos, width));

            // Update overlay width and slider position.
            overlay.style.width = `${newPos}px`;
            slider.style.left = `${newPos}px`;
        };

        slider.addEventListener('mousedown', () => {
            isPressed = true;
        });

        document.addEventListener('mousemove', moveSlider);
        document.addEventListener('mouseup', () => isPressed = false);
    });
});
