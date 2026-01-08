const video = document.getElementById('video');
const status = document.getElementById('status');

// Podgląd kamery w czasie rzeczywistym
navigator.mediaDevices.getUserMedia({ video: true })
.then(stream => {
    video.srcObject = stream;
})
.catch(err => {
    console.error("Camera error:", err);
});

function captureFrame() {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg');
}

function registerUser() {
    const imgData = captureFrame();
    fetch('/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: imgData })
    })
    .then(resp => resp.json())
    .then(data => status.innerText = data.message)
    .catch(err => status.innerText = "Registration error");
}

function loginUser() {
    const imgData = captureFrame();
    fetch('/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: imgData })
    })
    .then(resp => resp.json())
    .then(data => status.innerText = data.message)
    .catch(err => status.innerText = "Login error");
}

function sendImage(endpoint, canvas) {
    const dataURL = canvas.toDataURL('image/jpeg');
    fetch(endpoint, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({image: dataURL})
    })
    .then(resp => resp.json())
    .then(data => console.log(data));
}