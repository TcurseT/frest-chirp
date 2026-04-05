import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ChevronDown, Cpu, CheckCircle2 } from 'lucide-react'

const MODELS = [
  {
    id: 'bird_model_epoch5.pth',
    label: 'Epoch 5',
    tag: 'Early',
    desc: 'Early-stage checkpoint',
    accuracy: '~74%',
  },
  {
    id: 'bird_model_epoch8.pth',
    label: 'Epoch 8',
    tag: 'Best',
    desc: 'Optimal generalisation',
    accuracy: '~88%',
  },
]

export default function ModelSelector({ selectedModel, onModelChange }) {
  const [open, setOpen] = useState(false)
  const ref             = useRef(null)

  const selected = MODELS.find((m) => m.id === selectedModel) ?? MODELS[1]

  /* Close on outside click */
  useEffect(() => {
    const handler = (e) => {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  return (
    <div ref={ref} className="relative">
      {/* Trigger button */}
      <motion.button
        id="model-selector-btn"
        whileTap={{ scale: 0.96 }}
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-2 px-3 py-2 rounded-xl text-sm font-medium transition-colors duration-200"
        style={{
          background: open ? 'rgba(34,211,238,0.09)' : 'rgba(255,255,255,0.04)',
          border: `1px solid ${open ? 'rgba(34,211,238,0.35)' : 'rgba(255,255,255,0.09)'}`,
          color: '#94a3b8',
          backdropFilter: 'blur(12px)',
          boxShadow: open ? '0 0 20px rgba(34,211,238,0.1)' : 'none',
        }}
        aria-expanded={open}
        aria-haspopup="listbox"
      >
        <Cpu size={13} className="text-cyan-400 flex-shrink-0" />
        <span className="hidden sm:block text-slate-300 max-w-[170px] truncate font-mono text-xs">
          {selected.id}
        </span>
        <motion.span
          animate={{ rotate: open ? 180 : 0 }}
          transition={{ duration: 0.22, ease: 'easeInOut' }}
          className="flex-shrink-0"
        >
          <ChevronDown size={13} className="text-slate-500" />
        </motion.span>
      </motion.button>

      {/* Dropdown panel */}
      <AnimatePresence>
        {open && (
          <motion.div
            id="model-dropdown"
            role="listbox"
            initial={{ opacity: 0, y: -10, scale: 0.96 }}
            animate={{ opacity: 1, y: 0,   scale: 1 }}
            exit={{   opacity: 0, y: -10, scale: 0.96 }}
            transition={{ type: 'spring', stiffness: 420, damping: 28 }}
            className="absolute right-0 mt-2 w-80 rounded-2xl overflow-hidden z-[60]"
            style={{
              background: 'rgba(6,9,22,0.96)',
              border: '1px solid rgba(255,255,255,0.08)',
              backdropFilter: 'blur(28px)',
              boxShadow: '0 24px 64px rgba(0,0,0,0.55), 0 0 0 1px rgba(255,255,255,0.04) inset',
            }}
          >
            <div className="p-3">
              <p className="text-[10px] font-semibold text-slate-600 uppercase tracking-widest px-2 pb-2">
                Select Checkpoint
              </p>

              {MODELS.map((model) => {
                const isActive = selectedModel === model.id
                return (
                  <motion.button
                    key={model.id}
                    role="option"
                    aria-selected={isActive}
                    whileHover={{ x: 3 }}
                    transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                    onClick={() => { onModelChange(model.id); setOpen(false) }}
                    className="w-full flex items-center justify-between gap-3 px-3 py-3.5 rounded-xl text-left transition-colors duration-150"
                    style={{
                      background: isActive ? 'rgba(34,211,238,0.08)' : 'transparent',
                      border: `1px solid ${isActive ? 'rgba(34,211,238,0.2)' : 'transparent'}`,
                    }}
                    onMouseEnter={(e) => {
                      if (!isActive) e.currentTarget.style.background = 'rgba(255,255,255,0.04)'
                    }}
                    onMouseLeave={(e) => {
                      if (!isActive) e.currentTarget.style.background = 'transparent'
                    }}
                  >
                    {/* Left info */}
                    <div className="min-w-0">
                      <div className="flex items-center gap-2 mb-0.5">
                        <p className="text-slate-200 text-sm font-mono font-medium truncate">
                          {model.id}
                        </p>
                        <span
                          className="flex-shrink-0 text-[9px] font-bold px-1.5 py-0.5 rounded-full uppercase tracking-wider"
                          style={{
                            background: model.tag === 'Best'
                              ? 'rgba(34,211,238,0.15)'
                              : 'rgba(255,255,255,0.06)',
                            color: model.tag === 'Best' ? '#22d3ee' : '#64748b',
                            border: `1px solid ${model.tag === 'Best' ? 'rgba(34,211,238,0.3)' : 'rgba(255,255,255,0.08)'}`,
                          }}
                        >
                          {model.tag}
                        </span>
                      </div>
                      <p className="text-slate-600 text-xs">{model.desc} · Val {model.accuracy}</p>
                    </div>

                    {/* Checkmark */}
                    {isActive && (
                      <CheckCircle2 size={15} className="text-cyan-400 flex-shrink-0" />
                    )}
                  </motion.button>
                )
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
