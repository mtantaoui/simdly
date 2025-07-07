// A function to handle the parallax and fade animation.
const runHeroAnimation = () => {
  const heroImage = document.querySelector('.mdx-hero__image');
  
  // If the hero image element doesn't exist on the current page, do nothing.
  if (!heroImage) {
    return;
  }

  // Get the distance the user has scrolled down the page.
  const scrollPosition = window.scrollY;

  // Calculate the new opacity. It starts at 1 and fades to 0.
  // The '400' controls how fast it fades. A smaller number = faster fade.
  const opacity = 1 - Math.min(scrollPosition / 400, 1);

  // Calculate the vertical movement for the parallax effect.
  // The '0.4' controls the scroll speed. A larger number = faster movement.
  const translateY = scrollPosition * 0.4;

  // Apply the new styles to the image element.
  heroImage.style.opacity = opacity;
  heroImage.style.transform = `translateY(${translateY}px)`;
};

// Hook into Material for MkDocs's instant loading to handle page navigation.
if (typeof document$ !== "undefined") {
  document$.subscribe(() => {
    runHeroAnimation();
  });
}

// Attach the animation function to the window's scroll event.
// This makes the animation run live as the user scrolls.
window.addEventListener('scroll', runHeroAnimation, { passive: true });