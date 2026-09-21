let countInterval = null;

function log(message) {
  const logBox = document.getElementById('logOutput');
  const time = new Date().toLocaleTimeString();
  logBox.innerHTML = `> [${time}] ${message}`;
}

function setStatus(text, type = 'success') {
  const statusBadge = document.getElementById('systemStatus');
  if (!statusBadge) return;
  statusBadge.className = `badge bg-${type}-subtle text-${type} border border-${type}-subtle rounded-pill px-3 py-2`;
  statusBadge.innerHTML = `<i class="fa-solid fa-circle me-1 small"></i> ${text}`;
}

function startCountPolling() {
  if (countInterval) clearInterval(countInterval);

  countInterval = setInterval(async () => {
    try {
      const res = await fetch('/api/captures');
      const data = await res.json();

      if (data.status === 'success') {
        document.getElementById('faceCount').innerText = data.count;

        const tbody = document.getElementById('captureLogTable');
        if (data.logs && data.logs.length > 0) {
          tbody.innerHTML = data.logs.map((item, index) => {
            const isUnknown = item.name.toLowerCase() === 'unknown';
            const badgeClass = isUnknown ? 'bg-danger-subtle text-danger border-danger-subtle' : 'bg-success-subtle text-success border-success-subtle';

            return `
              <tr class="border-secondary">
                <td class="text-secondary">${index + 1}</td>
                <td class="fw-semibold text-white">${item.name}</td>
                <td class="text-info font-monospace small">${item.time}</td>
                <td>
                  <span class="badge ${badgeClass} border px-2 py-1">${isUnknown ? 'Unrecognized' : 'Recognized'}</span>
                </td>
              </tr>
            `;
          }).join('');
        }
      }
    } catch (err) {
      console.error('Error fetching capture logs:', err);
    }
  }, 1000);
}

function stopCountPolling() {
  if (countInterval) {
    clearInterval(countInterval);
    countInterval = null;
  }
}

async function startCollect() {
  const userName = document.getElementById('userNameInput').value.trim();
  if (!userName) {
    alert('Please enter a full name first!');
    return;
  }

  log(`Starting face collection for "${userName}"...`);
  setStatus('Collecting Data...', 'primary');

  const video = document.getElementById('videoStream');
  if (video) video.src = `/video_feed?t=${new Date().getTime()}`;

  try {
    const res = await fetch('/api/collect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: userName })
    });
    const data = await res.json();
    log(data.message);
  } catch (err) {
    log('Error starting face collection.');
  }
}

async function startTrain() {
  log('Training model, please wait...');
  setStatus('Training Model...', 'warning');

  try {
    const res = await fetch('/api/train', { method: 'POST' });
    const data = await res.json();
    log(data.message);
    setStatus('Model Trained', 'success');
  } catch (err) {
    log('Error training model.');
    setStatus('Error', 'danger');
  }
}

async function startRecognize() {
  log('Launching live face recognizer...');
  setStatus('System Active', 'success');

  const video = document.getElementById('videoStream');
  if (video) video.src = `/video_feed?t=${new Date().getTime()}`;

  startCountPolling();

  try {
    const res = await fetch('/api/recognize', { method: 'POST' });
    const data = await res.json();
    log(data.message);
  } catch (err) {
    log('Error starting recognizer.');
  }
}


async function stopSystem() {
  log('Shutting down camera hardware...');
  setStatus('Turning off...', 'warning');

  try {
    const res = await fetch('/api/stop', { method: 'POST' });
    const data = await res.json();
    log(data.message || 'Camera hardware powered off.');

    const video = document.getElementById('videoStream');
    if (video) video.removeAttribute('src');

    stopCountPolling();
    setStatus('Camera Offline', 'danger');
  } catch (err) {
    log('Failed to turn off camera.');
    console.error('Stop system error:', err);
  }
}

window.addEventListener('DOMContentLoaded', () => {
  startCountPolling();
});