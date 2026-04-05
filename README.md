# Frest Chirp

AI-powered Bird Species Identification via Audio Spectrograms and EfficientNet-B0.

## How to Run the Project

This project consists of two parts: a Python Flask backend (for the ML model) and a React/Vite frontend. You need **two separate terminals** to run them both.

### Step 1: Start the Backend (Flask API)

1. Open your terminal (PowerShell or Command Prompt).
2. Navigate to the root directory `Frest chirp`:
   ```powershell
   cd "C:\Users\anike\OneDrive\Desktop\Frest chirp"
   ```
3. Run the API using the existing virtual environment:
   ```powershell
   .\.venv\Scripts\python.exe api\app.py
   ```
   *You'll see output indicating the model is loading on your GPU (cuda) and the server running on `http://127.0.0.1:5000`.*

### Step 2: Start the Frontend (React + Vite)

1. Open a **second, new terminal**.
2. Navigate to the `app` folder inside the project:
   ```powershell
   cd "C:\Users\anike\OneDrive\Desktop\Frest chirp\app"
   ```
3. Start the Vite development server:
   ```powershell
   npm run dev
   ```
   *You'll see output indicating the server is ready, typically running on `http://localhost:5173`.*

### Step 3: Open the App

- Open your web browser and navigate to the URL provided by the frontend terminal (usually [http://localhost:5173](http://localhost:5173)).
- The frontend is already configured to automatically communicate with the Flask API behind the scenes.

## Troubleshooting

- **Microphone Access**: If trying to record live audio, ensure your browser has permission to access your microphone.
- **Port Conflicts**: Ensure ports `5000` and `5173` are not being used by other applications.
