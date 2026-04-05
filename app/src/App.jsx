import { useState, useRef } from 'react'
import Header from './components/Header'
import HeroCard from './components/HeroCard'
import ResultsSection from './components/ResultsSection'
import { buildMockResult } from './data/birds'
import bgVideo from './assets/Realistic_Bird_Video_Generation.mp4'

// Bird data / display metadata is loaded from src/data/birds.js
// Actual predictions come from the Flask API at /predict

const API_URL = '/predict'   // proxied by Vite to http://localhost:5000
const PHONE_APPEARS_TIME = 5 // Adjust this value to match the exact second the phone appears

export default function App() {
  const [selectedModel, setSelectedModel] = useState('bird_model_epoch8.pth')
  const [isRecording, setIsRecording] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [result, setResult] = useState(null)
  const [audioFileName, setAudioFileName] = useState(null)
  const [errorMsg, setErrorMsg] = useState(null)
  const [showUI, setShowUI] = useState(false)
  const resultsRef = useRef(null)
  const mediaRecRef = useRef(null)
  const chunksRef = useRef([])

  const handleTimeUpdate = (e) => {
    if (!showUI && e.target.currentTime >= PHONE_APPEARS_TIME) {
      setShowUI(true)
    }
  }

  /* ── Call real ML backend ── */
  const runInference = async (audioBlob, fileName) => {
    setIsProcessing(true)
    setResult(null)
    setErrorMsg(null)

    const startTime = Date.now()

    const form = new FormData()
    form.append('audio', audioBlob, fileName ?? 'recording.wav')
    form.append('model', selectedModel)

    try {
      const res = await fetch(API_URL, { method: 'POST', body: form })
      const data = await res.json()

      if (!res.ok) {
        throw new Error(data.error ?? `Server error ${res.status}`)
      }

      // Build display-enriched result object.
      // Real confidence + name come from the model; display metadata from birds.js.
      const enriched = buildMockResult(data.prediction)
      enriched.confidence = data.confidence   // override with real model confidence
      enriched.top5 = data.top5         // expose top-5 for optional display

      // Assure visualizer remains for at least 3 seconds
      const elapsed = Date.now() - startTime
      if (elapsed < 3000) {
        await new Promise(resolve => setTimeout(resolve, 3000 - elapsed))
      }

      setResult(enriched)
      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
      }, 120)
    } catch (err) {
      const elapsed = Date.now() - startTime
      if (elapsed < 3000) {
        await new Promise(resolve => setTimeout(resolve, 3000 - elapsed))
      }
      console.error('[predict]', err)
      setErrorMsg(err.message || 'Prediction failed — is the API server running?')
    } finally {
      setIsProcessing(false)
    }
  }

  const handleFileUpload = (file) => {
    if (!file) return
    setAudioFileName(file.name)
    setIsRecording(false)
    runInference(file, file.name)
  }

  /* ── Microphone recording ── */
  const handleRecord = async () => {
    if (isRecording) {
      // Stop recording → trigger inference
      mediaRecRef.current?.stop()
      setIsRecording(false)
    } else {
      setResult(null)
      setErrorMsg(null)
      setAudioFileName(null)
      chunksRef.current = []

      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
        const rec = new MediaRecorder(stream)
        mediaRecRef.current = rec

        rec.ondataavailable = (e) => {
          if (e.data.size > 0) chunksRef.current.push(e.data)
        }
        rec.onstop = () => {
          stream.getTracks().forEach(t => t.stop())
          const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
          setAudioFileName('live-recording.webm')
          runInference(blob, 'live-recording.webm')
        }

        rec.start()
        setIsRecording(true)
      } catch (err) {
        setErrorMsg('Microphone access denied: ' + err.message)
      }
    }
  }

  const handleReset = () => {
    setResult(null)
    setIsProcessing(false)
    setIsRecording(false)
    setAudioFileName(null)
    setErrorMsg(null)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  return (
    <div className="min-h-screen relative">
      {/* ── Video Background ── */}
      <video
        className="video-bg"
        autoPlay loop muted playsInline
        onTimeUpdate={handleTimeUpdate}
      >
        <source src={bgVideo} type="video/mp4" />
      </video>

      {/* ── Background UI overlays (fade in) ── */}
      <div className={`fixed inset-0 pointer-events-none transition-opacity duration-3000 ease-in z-[-10] ${showUI ? 'opacity-100' : 'opacity-0'}`}>
        <div className="absolute inset-0 bg-[#04060f]/60" />
        <div className="absolute inset-0 bg-grid opacity-100" />
        {/* Cyan radial glow – top centre */}
        <div
          className="absolute top-[-10%] left-1/2 -translate-x-1/2 w-[900px] h-[700px] rounded-full"
          style={{ background: 'radial-gradient(ellipse, rgba(34,211,238,0.08) 0%, transparent 65%)' }}
        />
        {/* Indigo glow – bottom right */}
        <div
          className="absolute bottom-[-5%] right-[-5%] w-[700px] h-[600px] rounded-full"
          style={{ background: 'radial-gradient(ellipse, rgba(129,140,248,0.06) 0%, transparent 65%)' }}
        />
        {/* Green tint – bottom left */}
        <div
          className="absolute bottom-[10%] left-[-5%] w-[500px] h-[400px] rounded-full"
          style={{ background: 'radial-gradient(ellipse, rgba(52,211,153,0.04) 0%, transparent 65%)' }}
        />
      </div>

      {/* ── Content (fade in) ── */}
      <div className={`relative z-10 transition-opacity duration-1000 ease-in-out ${showUI ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}>
        <Header
          selectedModel={selectedModel}
          onModelChange={setSelectedModel}
        />

        <main>
          {/* Hero */}
          <section className="min-h-screen flex items-center justify-center px-4 pt-24 pb-16">
            <HeroCard
              onFileUpload={handleFileUpload}
              onRecord={handleRecord}
              isRecording={isRecording}
              isProcessing={isProcessing}
              audioFileName={audioFileName}
            />

            {/* ── Error banner ── */}
            {errorMsg && (
              <div
                className="mt-6 w-full max-w-2xl mx-auto animation fade-in duration-3000 flex items-start gap-3 px-5 py-4 rounded-2xl"
                style={{
                  background: 'rgba(239,68,68,0.08)',
                  border: '1px solid rgba(239,68,68,0.25)',
                }}
              >
                <span className="text-red-400 text-sm flex-1">⚠️ {errorMsg}</span>
                <button
                  onClick={() => setErrorMsg(null)}
                  className="text-slate-600 hover:text-slate-300 text-xs ml-2 shrink-0"
                >✕</button>
              </div>
            )}

          </section>

          {/* Results */}
          <div ref={resultsRef}>
            <ResultsSection
              result={result}
              selectedModel={selectedModel}
              onReset={handleReset}
            />
          </div>
        </main>

        <footer className="py-10 text-center border-t border-white/[0.04]">
          <p className="text-slate-700 text-xs tracking-wide">
            AI Chirp Tracker &copy; {new Date().getFullYear()} &mdash; EfficientNet-B0 · BirdCLEF Dataset · STFT Spectrograms
          </p>
        </footer>
      </div>
    </div>
  )
}
