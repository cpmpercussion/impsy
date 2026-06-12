(function () {
  const STORAGE_KEY = 'impsy-theme';
  const order = ['auto', 'light', 'dark'];
  const icons = { auto: '◐', light: '☀', dark: '☾' };
  const labels = { auto: 'auto', light: 'light', dark: 'dark' };

  const button = document.getElementById('theme-toggle');
  if (!button) return;

  const iconEl = button.querySelector('[data-theme-icon]');
  const labelEl = button.querySelector('[data-theme-label]');
  const media = window.matchMedia('(prefers-color-scheme: dark)');

  const apply = (choice) => {
    const resolved = choice === 'auto'
      ? (media.matches ? 'dark' : 'light')
      : choice;
    document.documentElement.setAttribute('data-bs-theme', resolved);
    document.documentElement.dataset.themeChoice = choice;
    if (iconEl) iconEl.textContent = icons[choice];
    if (labelEl) labelEl.textContent = labels[choice];
    button.setAttribute('title', `Theme: ${labels[choice]} (click to cycle)`);
  };

  const initial = localStorage.getItem(STORAGE_KEY) || 'auto';
  apply(initial);

  button.addEventListener('click', () => {
    const current = document.documentElement.dataset.themeChoice || 'auto';
    const next = order[(order.indexOf(current) + 1) % order.length];
    localStorage.setItem(STORAGE_KEY, next);
    apply(next);
  });

  media.addEventListener('change', () => {
    if ((document.documentElement.dataset.themeChoice || 'auto') === 'auto') apply('auto');
  });
})();
