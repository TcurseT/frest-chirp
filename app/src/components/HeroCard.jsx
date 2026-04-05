import { useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Upload, Mic, MicOff, Zap } from 'lucide-react'
import AudioVisualizer, { PulsingMic } from './AudioVisualizer'

/* ── Static waveform decorative bars ─────────────────────── */
const WAVE_HEIGHTS = [4, 8, 14, 20, 28, 34, 40, 34, 28, 20, 14, 8, 4,
                      8, 14, 20, 28, 34, 28, 20, 14, 8, 4]

export default function HeroCard({
  onFileUpload,
  onRecord,
  isRecording,
  isProcessing,
  audioFileName,
}) {
  const fileRef             = useRef(null)
  const [dragging, setDragging] = useState(false)

  const handleDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer.files[0]
    if (f && (f.type.startsWith('audio/') || /\.(mp3|wav|ogg|flac)$/i.test(f.name))) {
      onFileUpload(f)
    }
  }

  return (
    <div className="w-full max-w-2xl mx-auto">
      {/* ── Headline ── */}
      <motion.div
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
        className="text-center mb-10"
      >
        {/* Pill badge */}
        <div
          className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full mb-6 text-xs font-semibold"
          style={{
            background: 'rgba(34,211,238,0.07)',
            border: '1px solid rgba(34,211,238,0.2)',
            color: '#22d3ee',
          }}
        >
          <Zap size={11} />
          Live AI Species Identification &middot; STFT + CNN
        </div>

        <h1
          className="text-5xl sm:text-6xl font-extrabold leading-[1.1] tracking-tight mb-4 text-white"
          style={{ fontFamily: 'Outfit, sans-serif' }}
        >
          Identify Birds<br />
          <span className="text-gradient-cyan-indigo">by Their Song</span>
        </h1>

        <p className="text-slate-400 text-lg max-w-md mx-auto leading-relaxed">
          Upload a recording or record live — our AI pinpoints the species in seconds
          using EfficientNet-B0 deep learning.
        </p>
      </motion.div>

      {/* ── Glass card ── */}
      <motion.div
        initial={{ opacity: 0, y: 36 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.75, delay: 0.18, type: 'spring', stiffness: 90, damping: 18 }}
        className="glass-card rounded-3xl p-8 relative overflow-hidden"
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        style={dragging ? {
          border: '1px solid rgba(34,211,238,0.5)',
          boxShadow: '0 0 0 1px rgba(34,211,238,0.2) inset, 0 32px 80px -16px rgba(0,0,0,0.65), 0 0 80px -20px rgba(34,211,238,0.2)',
        } : {}}
      >
        {/* Subtle inner glow on drag */}
        {dragging && (
          <div
            className="absolute inset-0 pointer-events-none rounded-3xl"
            style={{ background: 'rgba(34,211,238,0.04)' }}
          />
        )}

        <AnimatePresence mode="wait">
          {isProcessing ? (
            <ProcessingState key="proc" audioFileName={audioFileName} />
          ) : isRecording ? (
            <RecordingState key="rec" onStop={onRecord} />
          ) : (
            <IdleState
              key="idle"
              fileRef={fileRef}
              onRecord={onRecord}
              WAVE_HEIGHTS={WAVE_HEIGHTS}
              dragging={dragging}
            />
          )}
        </AnimatePresence>

        {/* Hidden file input */}
        <input
          ref={fileRef}
          id="audio-file-input"
          type="file"
          accept=".wav,.mp3,.ogg,.flac,audio/*"
          className="hidden"
          onChange={(e) => onFileUpload(e.target.files[0])}
        />
      </motion.div>
    </div>
  )
}

/* ─── IDLE state ─────────────────────────────────────────── */
function IdleState({ fileRef, onRecord, WAVE_HEIGHTS, dragging }) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0, scale: 0.97 }}
      transition={{ duration: 0.25 }}
    >
      {/* Decorative waveform */}
      <div className="flex items-center justify-center gap-[3px] mb-8" style={{ height: '44px' }}>
        {WAVE_HEIGHTS.map((h, i) => (
          <div
            key={i}
            className="rounded-full"
            style={{
              width: '4px',
              height: `${h}px`,
              background: 'linear-gradient(to top, rgba(34,211,238,0.5), rgba(129,140,248,0.3))',
              opacity: 0.45,
            }}
          />
        ))}
      </div>

      {/* Action buttons */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {/* Upload */}
        <ActionButton
          id="upload-audio-btn"
          color="cyan"
          icon={<Upload size={22} className="text-cyan-400" />}
          label="Upload Audio"
          sub=".wav  ·  .mp3  ·  .flac"
          onClick={() => fileRef.current?.click()}
        />

        {/* Record */}
        <ActionButton
          id="record-audio-btn"
          color="indigo"
          icon={<Mic size={22} className="text-indigo-400" />}
          label="Record Live Audio"
          sub="Real-time detection"
          onClick={onRecord}
        />
      </div>

      <p className="text-center text-slate-700 text-xs mt-5">
        You can also drag &amp; drop an audio file anywhere onto this card
      </p>
    </motion.div>
  )
}

/* ─── Shared action button ───────────────────────────────── */
function ActionButton({ id, icon, label, sub, onClick, color }) {
  const palette = {
    cyan: {
      base:   'rgba(34,211,238,0.05)',
      border: 'rgba(34,211,238,0.15)',
      hover:  'rgba(34,211,238,0.1)',
      hborder:'rgba(34,211,238,0.4)',
      shadow: '0 0 32px rgba(34,211,238,0.18), 0 0 60px rgba(34,211,238,0.06)',
      iconBg: 'linear-gradient(135deg,rgba(34,211,238,0.22),rgba(34,211,238,0.06))',
      iconBd: 'rgba(34,211,238,0.22)',
    },
    indigo: {
      base:   'rgba(129,140,248,0.05)',
      border: 'rgba(129,140,248,0.15)',
      hover:  'rgba(129,140,248,0.1)',
      hborder:'rgba(129,140,248,0.4)',
      shadow: '0 0 32px rgba(129,140,248,0.18), 0 0 60px rgba(129,140,248,0.06)',
      iconBg: 'linear-gradient(135deg,rgba(129,140,248,0.22),rgba(129,140,248,0.06))',
      iconBd: 'rgba(129,140,248,0.22)',
    },
  }
  const p = palette[color]

  return (
    <motion.button
      id={id}
      whileHover={{ scale: 1.025, y: -3 }}
      whileTap={{ scale: 0.975 }}
      transition={{ type: 'spring', stiffness: 400, damping: 22 }}
      onClick={onClick}
      className="flex flex-col items-center justify-center gap-3.5 p-8 rounded-2xl w-full text-center transition-all duration-300 relative overflow-hidden group"
      style={{ background: p.base, border: `1px solid ${p.border}` }}
      onMouseEnter={(e) => {
        e.currentTarget.style.background   = p.hover
        e.currentTarget.style.border       = `1px solid ${p.hborder}`
        e.currentTarget.style.boxShadow    = p.shadow
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background   = p.base
        e.currentTarget.style.border       = `1px solid ${p.border}`
        e.currentTarget.style.boxShadow    = 'none'
      }}
    >
      <div
        className="w-14 h-14 rounded-2xl flex items-center justify-center"
        style={{ background: p.iconBg, border: `1px solid ${p.iconBd}` }}
      >
        {icon}
      </div>
      <div>
        <p className="text-white font-semibold text-[15px]">{label}</p>
        <p className="text-slate-500 text-xs mt-1 font-mono">{sub}</p>
      </div>
    </motion.button>
  )
}

/* ─── RECORDING state ────────────────────────────────────── */
function RecordingState({ onStop }) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.94 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.94 }}
      transition={{ type: 'spring', stiffness: 280, damping: 22 }}
      className="flex flex-col items-center gap-7 py-6"
    >
      <PulsingMic />

      <div className="text-center">
        <div className="flex items-center justify-center gap-2 mb-1.5">
          <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
          <p className="text-white font-bold text-lg tracking-tight">Recording…</p>
        </div>
        <p className="text-slate-500 text-sm">Capturing live audio for bird identification</p>
      </div>

      {/* Live waveform */}
      <AudioVisualizer color="rgba(239,68,68,0.65)" count={28} />

      <motion.button
        id="stop-recording-btn"
        whileHover={{ scale: 1.04 }}
        whileTap={{ scale: 0.96 }}
        onClick={onStop}
        className="flex items-center gap-2 px-8 py-3 rounded-xl font-semibold text-sm text-white transition-all"
        style={{
          background: 'rgba(255,255,255,0.06)',
          border: '1px solid rgba(255,255,255,0.1)',
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.background = 'rgba(255,255,255,0.1)'
          e.currentTarget.style.border     = '1px solid rgba(255,255,255,0.2)'
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.background = 'rgba(255,255,255,0.06)'
          e.currentTarget.style.border     = '1px solid rgba(255,255,255,0.1)'
        }}
      >
        <MicOff size={16} />
        Stop &amp; Identify
      </motion.button>
    </motion.div>
  )
}

/* ─── PROCESSING state ───────────────────────────────────── */
function ProcessingState({ audioFileName }) {
  const steps = ['STFT Transform', 'Spectrogram', 'CNN Forward Pass', 'Classification']

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.95 }}
      transition={{ duration: 0.3 }}
      className="flex flex-col items-center gap-6 py-4"
    >
      {/* Animated spectrogram preview */}
      <div
        className="relative w-full max-w-sm h-28 rounded-2xl overflow-hidden"
        style={{
          background: 'rgba(0,0,0,0.35)',
          border: '1px solid rgba(255,255,255,0.06)',
        }}
      >
        <div className="absolute inset-0 flex items-end gap-[2px] p-2">
          {Array.from({ length: 44 }).map((_, i) => (
            <motion.div
              key={i}
              className="flex-1 rounded-[2px]"
              style={{
                background: `hsl(${175 + i * 3}, 75%, ${45 + (i % 4) * 5}%)`,
              }}
              animate={{
                height: [
                  `${25 + Math.random() * 45}%`,
                  `${40 + Math.random() * 45}%`,
                  `${20 + Math.random() * 35}%`,
                ],
                opacity: [0.6, 1, 0.6],
              }}
              transition={{
                duration: 0.45 + (i % 7) * 0.07,
                repeat: Infinity,
                repeatType: 'reverse',
                delay: i * 0.015,
                ease: 'easeInOut',
              }}
            />
          ))}
        </div>

        {/* Scanning line */}
        <motion.div
          className="absolute top-0 bottom-0 w-[2px] pointer-events-none"
          style={{ background: 'rgba(34,211,238,0.6)', boxShadow: '0 0 8px rgba(34,211,238,0.8)' }}
          animate={{ left: ['5%', '92%', '5%'] }}
          transition={{ duration: 2.4, repeat: Infinity, ease: 'linear' }}
        />

        {/* Label */}
        <div className="absolute inset-0 flex items-start justify-end p-2">
          <span
            className="text-[10px] font-medium px-2 py-1 rounded-md"
            style={{
              background: 'rgba(0,0,0,0.65)',
              color: '#22d3ee',
              border: '1px solid rgba(34,211,238,0.18)',
            }}
          >
            STFT Spectrogram
          </span>
        </div>
      </div>

      <div className="text-center">
        <p className="text-white font-bold text-lg mb-1">Analysing Audio</p>
        {audioFileName && (
          <p className="text-slate-500 text-xs font-mono mb-1 truncate max-w-xs">{audioFileName}</p>
        )}
        <p className="text-slate-600 text-xs">Running EfficientNet-B0 inference…</p>
      </div>

      {/* Pipeline steps */}
      <div className="flex flex-wrap items-center justify-center gap-1 text-[11px] text-slate-600">
        {steps.map((s, i) => (
          <span key={s} className="flex items-center gap-1">
            <motion.span
              animate={{ opacity: [0.3, 1, 0.3] }}
              transition={{ duration: 1.4, delay: i * 0.38, repeat: Infinity }}
              style={{ color: '#94a3b8' }}
            >
              {s}
            </motion.span>
            {i < steps.length - 1 && (
              <span className="text-slate-800 select-none">›</span>
            )}
          </span>
        ))}
      </div>
    </motion.div>
  )
}
