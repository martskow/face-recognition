// Globalne zmienne dla multimediów
const video = document.getElementById('video');
const status = document.getElementById('status');

let mediaRecorder;
let audioChunks = [];
let latestAudioBase64 = null;

// =========================================================================
// 1. OBSŁUGA KAMERY (Bez automatycznego włączania mikrofonu na starcie!)
// =========================================================================
if (video) {
    // Żądamy tylko wideo, aby nie wymuszać ikony mikrofonu przy wejściu na stronę
    navigator.mediaDevices.getUserMedia({ video: true, audio: false })
    .then(stream => {
        video.srcObject = stream;
    })
    .catch(err => {
        console.error("Błąd dostępu do kamery:", err);
    });
}

function toggleVoiceRecording() {
    const btn = document.getElementById('record-btn');
    if (!mediaRecorder) return alert("System nie wykrył mikrofonu lub brak uprawnień.");

    if (mediaRecorder.state === "inactive") {
        audioChunks = [];
        mediaRecorder.start();
        btn.innerText = "Nagrywanie... Kliknij, aby zatrzymać";
        btn.style.background = "#dc3545";
    } else {
        mediaRecorder.stop();
        btn.innerText = "Głos nagrany! Kliknij, aby powtórzyć";
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
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));

    if (event && event.currentTarget) {
        event.currentTarget.classList.add('active');
    }

    const targetContent = document.getElementById(`tab-${tabName}`);
    if (targetContent) {
        targetContent.classList.add('active');
    }

    if (tabName === 'history') {
        loadLoginHistory();
    }

    if (tabName === 'tab-biometrics') {
        if (typeof startCamera === 'function') {
            startCamera();
        } else if (typeof initCamera === 'function') {
            initCamera();
        }
    }
}

// =========================================================================
// 3. IMPLEMENTACJA AKCJI PANELU UŻYTKOWNIKA (API)
// =========================================================================

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
        const userDisplay = document.getElementById('user-display-name');
        if (userDisplay) userDisplay.innerText = fn;
    })
    .catch(err => {
        statusTxt.style.color = "red";
        statusTxt.innerText = "Nie udało się zaktualizować profilu.";
    });
}

function loadLoginHistory() {
    const tbody = document.getElementById('history-table-body');
    if (!tbody) return;

    fetch('/api/user/history')
    .then(resp => resp.json())
    .then(data => {
        tbody.innerHTML = "";
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

function initBiometricsTab() {
    const imgFeed = document.getElementById("biometricsVideoFeed");
    if (imgFeed) {
        imgFeed.src = "/video?ts=" + Date.now();
    }
}

setInterval(() => {
    const el = document.getElementById("biometrics_face_status");
    if (!el) return;

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

function getBiometricsFrameBase64() {
    const videoFeed = document.getElementById('biometricsVideoFeed');
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

function reinitFaceOnly() {
    const statusTxt = document.getElementById('biometrics-status');
    const base64Image = getBiometricsFrameBase64();

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
        body: JSON.stringify({ image: base64Image })
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
                        body: JSON.stringify({ audio: base64Audio })
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
// 4. NOWA LOGIKA LOGOWANIA (Z warunkowym sprawdzaniem i nagrywaniem głosu)
// =========================================================================
function loginUser() {
    const emailEl = document.getElementById('email');
    const passwordEl = document.getElementById('password');
    const statusTxt = document.getElementById('status');

    if (!emailEl || !passwordEl) {
        console.error("Nie znaleziono pól formularza logowania (email/password) w drzewie DOM.");
        return;
    }

    const email = emailEl.value;
    const password = passwordEl.value;

    if (!email || !password) {
        if (statusTxt) statusTxt.innerText = "Proszę uzupełnić adres e-mail oraz hasło.";
        return;
    }

    if (statusTxt) {
        statusTxt.innerText = "Sprawdzanie konfiguracji konta...";
        statusTxt.style.color = "orange";
    }

    // KROK 1: Szybkie sprawdzenie w bazie danych
    fetch('/api/login-check', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: email, password: password })
    })
    .then(resp => {
        if (!resp.ok) {
            return resp.json().then(err => { throw new Error(err.message || "Błędne dane logowania"); });
        }
        return resp.json();
    })
    .then(preCheck => {
        if (!preCheck.valid) {
            throw new Error(preCheck.message || "Autoryzacja odrzucona");
        }

        // KROK 2: Przechwytujemy zdjęcie twarzy
        const faceImage = captureFrame();
        if (!faceImage) {
            throw new Error("Nie można przechwycić obrazu z kamery. Upewnij się, że dasz do niej dostęp.");
        }

        // KROK 3: Sprawdzamy flagę pobraną z bazy
        if (preCheck.require_voice) {
            // Użytkownik MA WŁĄCZONY suwak -> dopiero tutaj żądamy dostępu do AUDIO i nagrywamy
            if (statusTxt) {
                statusTxt.innerText = "Wymagana autoryzacja głosem. Powiedz coś teraz (3s)...";
                statusTxt.style.color = "purple";
            }

            navigator.mediaDevices.getUserMedia({ audio: true })
            .then(audioStream => {
                const loginRecorder = new MediaRecorder(audioStream);
                const chunks = [];

                loginRecorder.ondataavailable = e => chunks.push(e.data);
                loginRecorder.onstop = () => {
                    const audioBlob = new Blob(chunks, { type: 'audio/webm' });
                    const reader = new FileReader();
                    reader.readAsDataURL(audioBlob);
                    reader.onloadend = () => {
                        const base64Audio = reader.result;
                        sendFinalLoginRequest(email, password, faceImage, base64Audio, statusTxt);
                    };
                    audioStream.getTracks().forEach(track => track.stop());
                };

                loginRecorder.start();
                setTimeout(() => {
                    loginRecorder.stop();
                }, 3000);
            })
            .catch(err => {
                if (statusTxt) {
                    statusTxt.innerText = "Błąd mikrofonu: " + err.message;
                    statusTxt.style.color = "red";
                }
            });

        } else {
            // Użytkownik WYŁĄCZYŁ suwak -> Całkowicie pomijamy mikrofon
            if (statusTxt) {
                statusTxt.innerText = "Autoryzacja uproszczona (Tylko Twarz). Przetwarzanie...";
                statusTxt.style.color = "blue";
            }
            sendFinalLoginRequest(email, password, faceImage, null, statusTxt);
        }
    })
    .catch(err => {
        if (statusTxt) {
            statusTxt.innerText = err.message;
            statusTxt.style.color = "red";
        }
    });
}

function sendFinalLoginRequest(email, password, imageBase64, audioBase64, statusElement) {
    const payload = {
        email: email,
        password: password,
        image: imageBase64,
        audio: audioBase64
    };

    fetch('/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
    .then(resp => {
        if (!resp.ok) {
            return resp.json().then(err => { throw new Error(err.message || "Błąd uwierzytelniania biometrycznego."); });
        }
        return resp.json();
    })
    .then(data => {
        if (statusElement) {
            statusElement.innerText = data.message;
            statusElement.style.color = "green";
        }
        if (data.redirect) {
            window.location.href = data.redirect;
        }
    })
    .catch(err => {
        if (statusElement) {
            statusElement.innerText = err.message;
            statusElement.style.color = "red";
        }
    });
}

window.addEventListener("load", () => {
    initBiometricsTab();
});