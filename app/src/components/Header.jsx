import { Bird } from 'lucide-react'
import ModelSelector from './ModelSelector'

export default function Header({ selectedModel, onModelChange }) {
  return (
    <header className="fixed top-0 left-0 right-0 z-50">
      <div
        style={{
          background: 'rgba(4,6,15,0.75)',
          backdropFilter: 'blur(24px) saturate(180%)',
          WebkitBackdropFilter: 'blur(24px) saturate(180%)',
          borderBottom: '1px solid rgba(255,255,255,0.05)',
        }}
      >
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between gap-4">
          {/* Logo */}
          <div className="flex items-center gap-3 flex-shrink-0">
            <div
              className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0"
              style={{
                background: 'linear-gradient(135deg,rgba(34,211,238,0.18),rgba(129,140,248,0.18))',
                border: '1px solid rgba(34,211,238,0.28)',
                boxShadow: '0 0 20px rgba(34,211,238,0.15)',
              }}
            >
              <Bird size={18} className="text-cyan-400" />
            </div>
            <div>
              <p
                className="text-white font-semibold text-[15px] leading-tight tracking-tight"
                style={{ fontFamily: 'Outfit, sans-serif' }}
              >
                AI Chirp Tracker
              </p>
              <p className="text-slate-600 text-[11px] leading-none mt-0.5 font-medium tracking-wide">
                Bird Species Detector
              </p>
            </div>
          </div>

          {/* Right side */}
          <div className="flex items-center gap-3">
            {/* Subtle badge */}
            <div
              className="hidden md:flex items-center gap-1.5 px-3 py-1.5 rounded-full text-[11px] font-medium"
              style={{
                background: 'rgba(52,211,153,0.08)',
                border: '1px solid rgba(52,211,153,0.18)',
                color: '#34d399',
              }}
            >
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
              EfficientNet-B0 · 88% Val Accuracy
            </div>

            <ModelSelector selectedModel={selectedModel} onModelChange={onModelChange} />
          </div>
        </div>
      </div>
    </header>
  )
}
