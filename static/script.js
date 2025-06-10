const form = document.getElementById('cubeForm');
const resultImg = document.getElementById('resultImg');
const downloadLink = document.getElementById('downloadLink');
const spinnerOverlay = document.getElementById('spinnerOverlay');
const submitButton = form.querySelector('button[type="submit"]');

form.addEventListener('submit', async (e) => {
  e.preventDefault();

  // Скрываем старый результат и ссылку
  resultImg.style.display = 'none';
  downloadLink.style.display = 'none';

  // Показываем спиннер и блокируем кнопку
  spinnerOverlay.style.display = 'flex';
  submitButton.disabled = true;

  const formData = new FormData(form);

  try {
    // Отправляем POST-запрос на /stitch
    const response = await fetch('/stitch', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Ошибка сервера: ${response.status} — ${text}`);
    }

    // Получаем JPG в виде Blob и создаём URL
    const blob = await response.blob();
    const imgURL = URL.createObjectURL(blob);

    // Скрываем спиннер, показываем картинку и ссылку
    spinnerOverlay.style.display = 'none';
    submitButton.disabled = false;

    resultImg.src = imgURL;
    resultImg.style.display = 'block';

    downloadLink.href = imgURL;
    downloadLink.style.display = 'inline';
  } catch (err) {
    // При ошибке тоже скрываем спиннер и разблокируем кнопку
    spinnerOverlay.style.display = 'none';
    submitButton.disabled = false;

    alert('Не удалось получить панораму: ' + err.message);
  }
});
