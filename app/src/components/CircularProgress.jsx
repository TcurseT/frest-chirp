import { motion } from 'framer-motion'

/**
 * Animated SVG circular progress ring.
 * @param {number} value   – 0-100
 * @param {number} size    – diameter in px (default 88)
 * @param {number} stroke  – stroke width (default 7)
 */
export default function CircularProgress({ value = 0, size = 88, stroke = 7 }) {
  const r    = (size - stroke) / 2
  const circ = 2 * Math.PI * r
  const off  = circ - (value / 100) * circ

  return (
    <div className="relative flex-shrink-0" style={{ width: size, height: size }}>
      <svg width={size} height={size} style={{ transform: 'rotate(-90deg)' }}>
        <defs>
          <linearGradient id="cg" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%"   stopColor="#22d3ee" />
            <stop offset="100%" stopColor="#818cf8" />
          </linearGradient>
        </defs>

        {/* Track */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke="rgba(255,255,255,0.06)"
          strokeWidth={stroke}
        />

        {/* Progress arc */}
        <motion.circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke="url(#cg)"
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={circ}
          initial={{ strokeDashoffset: circ }}
          animate={{ strokeDashoffset: off }}
          transition={{ duration: 1.6, ease: [0.16, 1, 0.3, 1], delay: 0.4 }}
          style={{ filter: 'drop-shadow(0 0 6px rgba(34,211,238,0.5))' }}
        />
      </svg>

      {/* Centre label */}
      <div className="absolute inset-0 flex items-center justify-center">
        <motion.span
          className="text-sm font-bold text-white tabular-nums"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8 }}
        >
          {Math.round(value)}%
        </motion.span>
      </div>
    </div>
  )
}
