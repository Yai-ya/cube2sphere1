const form = document.getElementById('cubeForm');
const resultImg = document.getElementById('resultImg');
const downloadLink = document.getElementById('downloadLink');

form.addEventListener('submit', async (e) => {
  e.preventDefault();

  // Прячем предыдущие изображения/ссылки (если были)
  resultImg.style.display = 'none';
  downloadLink.style.display = 'none';

  const formData = new FormData(form);

  try {
    // Делаем запрос на тот же домен: POST /stitch
    const response = await fetch('/stitch', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Ошибка сервера: ${response.status} ${text}`);
    }

    // Читаем ответ как Blob (JPG)
    const blob = await response.blob();
    const imgURL = URL.createObjectURL(blob);

    // Показываем картинку на странице
    resultImg.src = imgURL;
    resultImg.style.display = 'block';

    // Настраиваем ссылку для скачивания
    downloadLink.href = imgURL;
    downloadLink.style.display = 'inline';
  } catch (err) {
    alert('Не удалось получить панораму: ' + err.message);
  }
});
