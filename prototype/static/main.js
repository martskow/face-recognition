// Globalne zmienne dla multimediów
const video = document.getElementById('video');
const status = document.getElementById('status');

let mediaRecorder;
let audioChunks = [];
let latestAudioBase64 = null;

// =========================================================================
// 1. OBSŁUGA KAMERY I MIKROFONU (Zintegrowana z zakładką Biometria)
// =========================================================================
if (video) {
    navigator.mediaDevices.getUserMedia({ video: true, audio: true })
    .then(stream => {
        video.srcObject = stream;

        // Konfiguracja nagrywania dźwięku (Web Audio API)
        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.ondataavailable = event => audioChunks.push(event.data);
        mediaRecorder.onstop = () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            const reader = new FileReader();
            reader.readAsDataURL(audioBlob);
            reader.onloadend = () => {
                latestAudioBase64 = reader.result;
                console.log("Audio zostało pomyślnie przekonwertowane do Base64.");
            };
        };
    })
    .catch(err => {
        console.error("Błąd dostępu do kamery/mikrofonu:", err);
    });
}

function toggleVoiceRecording() {
    const btn = document.getElementById('record-btn');
    if (!mediaRecorder) return alert("System nie wykrył mikrofonu lub brak uprawnień.");

    if (mediaRecorder.state === "inactive") {
        audioChunks = [];
        mediaRecorder.start();
        btn.innerText = "🛑 Nagrywanie... Kliknij, aby zatrzymać";
        btn.style.background = "#dc3545";
    } else {
        mediaRecorder.stop();
        btn.innerText = "🎤 Głos nagrany! Kliknij, aby powtórzyć";
        btn.style.background = "#28a745";
    }
}

function captureFrame() {
    if (!video) return null;
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg');
}

// =========================================================================
// 2. LOGIKA PRZEŁĄCZANIA ZAKŁADEK (TABS)
// =========================================================================
function switchTab(tabName) {
    // 1. Ukryj wszystkie zawartości zakładek i usuń klasę active z przycisków
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));

    // 2. Aktywuj kliknięty przycisk i pokaż powiązaną zawartość
    // Używamy currentTarget, aby upewnić się, że łapiemy właściwy element przycisku
    if (event && event.currentTarget) {
        event.currentTarget.classList.add('active');
    }

    const targetContent = document.getElementById(`tab-${tabName}`);
    if (targetContent) {
        targetContent.classList.add('active');
    }

    // 3. Akcja specjalna: Jeśli wchodzimy do historii, pobierz świeże dane z bazy
    if (tabName === 'history') {
        loadLoginHistory();
    }

    if (tabName === 'tab-biometrics') {
        // Wywołaj Twoją oryginalną funkcję, która odpala kamerę
        // i podczepia ją pod tag <video id="video">
        if (typeof startCamera === 'function') {
            startCamera();
        } else if (typeof initCamera === 'function') {
            initCamera();
        }
    }
}

// =========================================================================
// 3. Z3: IMPLEMENTACJA AKCJI PANELU UŻYTKOWNIKA (API)
// =========================================================================

// Aktualizacja danych profilu (Imię, Nazwisko, Hasło)
function updateProfile() {
    const fn = document.getElementById('edit-firstname').value;
    const ln = document.getElementById('edit-lastname').value;
    const pwd = document.getElementById('edit-password').value;
    const statusTxt = document.getElementById('profile-status');

    statusTxt.style.color = "orange";
    statusTxt.innerText = "Trwa zapisywanie...";

    fetch('/api/user/profile', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ first_name: fn, last_name: ln, password: pwd })
    })
    .then(resp => {
        if (!resp.ok) throw new Error("Błąd serwera");
        return resp.json();
    })
    .then(data => {
        statusTxt.style.color = "green";
        statusTxt.innerText = data.message;
        // Dynamicznie zaktualizuj imię na powitaniu u góry strony
        const userDisplay = document.getElementById('user-display-name');
        if (userDisplay) userDisplay.innerText = fn;
    })
    .catch(err => {
        statusTxt.style.color = "red";
        statusTxt.innerText = "Nie udało się zaktualizować profilu.";
    });
}

// Pobieranie i renderowanie historii logowań użytkownika
function loadLoginHistory() {
    const tbody = document.getElementById('history-table-body');
    if (!tbody) return;

    fetch('/api/user/history')
    .then(resp => resp.json())
    .then(data => {
        tbody.innerHTML = ""; // Wyczyszczenie starej tabeli
        if (data.length === 0) {
            tbody.innerHTML = "<tr><td colspan='3' style='text-align:center; color:#888;'>Brak historii logowań</td></tr>";
            return;
        }
        data.forEach(row => {
            const statusClass = row.status.toLowerCase().includes('success') || row.status.toLowerCase().includes('passed') ? 'status-success' : 'status-failed';
            tbody.innerHTML += `
                <tr>
                    <td>${row.timestamp}</td>
                    <td>${row.ip}</td>
                    <td class="${statusClass}">${row.status}</td>
                </tr>
            `;
        });
    })
    .catch(err => {
        tbody.innerHTML = "<tr><td colspan='3' style='text-align:center; color:red;'>Błąd ładowania danych</td></tr>";
    });
}

// 1. URUCHOMIENIE STRUMIENIA I SPRAWDZANIA TWARZY PO WEJŚCIU NA DASHBOARD
function initBiometricsTab() {
    // Podłączenie strumienia z Flaska pod obrazek w zakładce biometrii
    const imgFeed = document.getElementById("biometricsVideoFeed");
    if (imgFeed) {
        imgFeed.src = "/video?ts=" + Date.now();
    }
}

// Sprawdzanie statusu twarzy w zakładce biometrii (co 500ms)
setInterval(() => {
    const el = document.getElementById("biometrics_face_status");
    if (!el) return; // wykonaj tylko jeśli zakładka istnieje/jest widoczna

    fetch("/face_status")
        .then(r => r.json())
        .then(data => {
            if (data.face_detected) {
                el.textContent = "Face detected";
                el.style.color = "green";
            } else {
                el.textContent = "Face not detected yet";
                el.style.color = "red";
            }
        }).catch(err => {});
}, 500);


// 2. AKTUALIZACJA SAMEJ TWARZY (Wysyła tylko pusty sygnał, bo backend sam bierze klatkę przez camera.get_frame())
// Pomocnicza funkcja pobierająca obraz z elementu graficznego (identycznie jak w rejestracji)
function getBiometricsFrameBase64() {
    const videoFeed = document.getElementById('biometricsVideoFeed');
    // Tworzymy dynamicznie ukryty canvas o wymiarach takich jak w rejestracji
    const canvas = document.createElement('canvas');
    canvas.width = 160;
    canvas.height = 160;
    const ctx = canvas.getContext('2d');

    try {
        ctx.drawImage(videoFeed, 0, 0, canvas.width, canvas.height);
        return canvas.toDataURL('image/png');
    } catch (e) {
        console.error("Błąd przechwytywania klatki canvas:", e);
        return null;
    }
}

// Zaktualizowana funkcja wysyłająca PRAWDZIWY Base64 twarzy
function reinitFaceOnly() {
    const statusTxt = document.getElementById('biometrics-status');
    const base64Image = getBiometricsFrameBase64(); // Przechwytujemy klatkę!

    if (!base64Image) {
        statusTxt.style.color = "red";
        statusTxt.innerText = "Błąd: Nie można przechwycić obrazu ze strumienia wideo.";
        return;
    }

    statusTxt.style.color = "orange";
    statusTxt.innerText = "Przetwarzanie i zapisywanie nowego profilu twarzy...";

    fetch('/api/user/reinit_biometrics', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: base64Image }) // Wysyłamy autentyczny Base64
    })
    .then(resp => {
        if (!resp.ok) return resp.json().then(err => { throw new Error(err.message); });
        return resp.json();
    })
    .then(data => {
        statusTxt.style.color = "green";
        statusTxt.innerText = data.message;
    })
    .catch(err => {
        statusTxt.style.color = "red";
        statusTxt.innerText = err.message || "Błąd komunikacji z serwerem.";
    });
}


// 3. AKTUALIZACJA SAMEGO GŁOSU (Dokładnie tak jak w login.html - nagrywanie 3s i wysyłka)
function reinitVoiceWithRecording() {
    const statusTxt = document.getElementById('biometrics-status');
    const voiceBtn = document.getElementById('voiceBtn');

    statusTxt.innerText = "Żądanie dostępu do mikrofonu...";
    statusTxt.style.color = "orange";

    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            const mediaRecorder = new MediaRecorder(stream);
            const audioChunks = [];

            mediaRecorder.addEventListener("dataavailable", event => {
                audioChunks.push(event.data);
            });

            mediaRecorder.addEventListener("stop", () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                const reader = new FileReader();
                reader.readAsDataURL(audioBlob);

                reader.onloadend = () => {
                    const base64Audio = reader.result;

                    statusTxt.innerText = "Przesyłanie nowego wzorca mowy do bazy danych...";
                    statusTxt.style.color = "blue";

                    fetch('/api/user/reinit_biometrics', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ audio: base64Audio }) // Wysyłamy wygenerowane audio
                    })
                    .then(res => res.json())
                    .then(data => {
                        statusTxt.innerText = data.message;
                        statusTxt.style.color = "green";
                        voiceBtn.style.background = "#4CAF50";
                        voiceBtn.innerText = "Nagraj i zaktualizuj Głos";
                    })
                    .catch(err => {
                        statusTxt.innerText = "Błąd: " + err.message;
                        statusTxt.style.color = "red";
                    });
                };
            });

            // Start 3-sekundowego nagrywania
            mediaRecorder.start();
            statusTxt.innerText = "Nagrywanie głosu... Powiedz coś teraz (3s).";
            statusTxt.style.color = "purple";
            voiceBtn.innerText = "NAGRYWANIE...";

            setTimeout(() => {
                mediaRecorder.stop();
                stream.getTracks().forEach(track => track.stop());
            }, 3000);

        })
        .catch(err => {
            statusTxt.innerText = "Błąd mikrofonu: " + err.message;
            statusTxt.style.color = "red";
        });
}

// =========================================================================
// 4. STARE KOMENDY DO LOGOWANIA I REJESTRACJI (Zachowane dla kompatybilności)
// =========================================================================
function registerUser() {
    const imgData = captureFrame();
    fetch('/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: imgData })
    })
    .then(resp => resp.json())
    .then(data => { if(status) status.innerText = data.message; })
    .catch(() => { if(status) status.innerText = "Registration error"; });
}

function loginUser() {
    const imgData = captureFrame();
    fetch('/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: imgData })
    })
    .then(resp => resp.json())
    .then(data => {
        if(status) status.innerText = data.message;
        if (data.redirect) window.location.href = data.redirect;
    })
    .catch(() => { if(status) status.innerText = "Login error"; });
}

window.addEventListener("load", () => {
    initBiometricsTab();
});