import { motion } from 'framer-motion'
import { Mic } from 'lucide-react'

/**
 * Animated live audio visualizer bar group.
 * Used in the recording state to show live waveform feedback.
 */
export default function AudioVisualizer({ color = 'rgba(239,68,68,0.7)', count = 26 }) {
  const bars = Array.from({ length: count }, (_, i) => {
    // Stagger heights to create a natural waveform envelope
    const centre = count / 2
    const dist   = Math.abs(i - centre) / centre     // 0 at edges, 0 at centre
    const base   = 8 + (1 - dist) * 18               // taller in the middle
    return base
  })

  return (
    <div className="flex items-center justify-center gap-[3px]" style={{ height: '48px' }}>
      {bars.map((base, i) => (
        <motion.div
          key={i}
          className="rounded-full"
          style={{
            width: '3px',
            background: color,
            minHeight: '4px',
          }}
          animate={{
            height: [
              `${base * 0.5}px`,
              `${base * (0.8 + Math.random() * 0.8)}px`,
              `${base * 0.6}px`,
              `${base * (1 + Math.random())}px`,
              `${base * 0.5}px`,
            ],
          }}
          transition={{
            duration: 0.5 + (i % 5) * 0.1,
            repeat: Infinity,
            repeatType: 'loop',
            ease: 'easeInOut',
            delay: i * 0.035,
          }}
        />
      ))}
    </div>
  )
}

/* ─── Pulsing mic icon for reuse ─────────────────────────── */
export function PulsingMic() {
  return (
    <div className="relative flex items-center justify-center w-28 h-28">
      {/* Outer ring B */}
      <div
        className="ring-b absolute inset-0 rounded-full"
        style={{
          background: 'rgba(239,68,68,0.04)',
          border: '1px solid rgba(239,68,68,0.12)',
        }}
      />
      {/* Outer ring A */}
      <div
        className="ring-a absolute inset-[14px] rounded-full"
        style={{
          background: 'rgba(239,68,68,0.07)',
          border: '1px solid rgba(239,68,68,0.22)',
        }}
      />
      {/* Core button */}
      <div
        className="relative z-10 w-16 h-16 rounded-full flex items-center justify-center glow-red"
        style={{
          background: 'linear-gradient(145deg, #ef4444, #b91c1c)',
        }}
      >
        <Mic size={26} className="text-white" />
      </div>
    </div>
  )
}
