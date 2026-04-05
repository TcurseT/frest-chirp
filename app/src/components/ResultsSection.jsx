import { motion } from 'framer-motion'
import { MapPin, Leaf, Bird, Sparkles, RotateCcw, Info } from 'lucide-react'
import CircularProgress from './CircularProgress'

export default function ResultsSection({ result, selectedModel, onReset }) {

  if (!result) return null

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.1, delayChildren: 0.1 },
    },
  }

  const itemVariants = {
    hidden:  { opacity: 0, y: 28 },
    visible: { opacity: 1, y: 0, transition: { type: 'spring', stiffness: 90, damping: 18 } },
  }

  return (
    <section className="min-h-[80vh] flex flex-col items-center justify-center px-4 py-24">
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="w-full max-w-4xl"
      >
        {/* ── Section heading ── */}
        <motion.div variants={itemVariants} className="text-center mb-10">
          <div
            className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-xs font-semibold mb-4"
            style={{
              background: 'rgba(52,211,153,0.07)',
              border: '1px solid rgba(52,211,153,0.2)',
              color: '#34d399',
            }}
          >
            <Sparkles size={11} />
            Classification Result
          </div>
          <h2
            className="text-3xl sm:text-4xl font-bold text-white"
            style={{ fontFamily: 'Outfit, sans-serif' }}
          >
            Species Identified
          </h2>
          <p className="text-slate-600 text-xs mt-2 font-mono">
            Model: {selectedModel}
          </p>
        </motion.div>

        {/* ── Main result card ── */}
        <motion.div variants={itemVariants} className="glass-card rounded-3xl overflow-hidden">
          <div className="grid grid-cols-1 md:grid-cols-2">
            {/* ── LEFT: Bird visual + name ── */}
            <div
              className="relative flex flex-col items-center justify-center gap-6 p-10"
              style={{ borderRight: '1px solid rgba(255,255,255,0.04)' }}
            >
              {/* Glow orb behind image */}
              <div
                className="absolute inset-0 pointer-events-none"
                style={{
                  background:
                    'radial-gradient(ellipse at 50% 50%, rgba(52,211,153,0.08) 0%, transparent 70%)',
                }}
              />

              {/* Bird image placeholder */}
              <motion.div
                initial={{ scale: 0.78, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ type: 'spring', stiffness: 100, damping: 16, delay: 0.25 }}
                className="relative z-10 w-52 h-52 rounded-3xl flex flex-col items-center justify-center overflow-hidden"
                style={{
                  background:
                    'linear-gradient(145deg, rgba(52,211,153,0.1), rgba(34,211,238,0.05))',
                  border: '1px solid rgba(52,211,153,0.14)',
                  boxShadow: '0 0 60px rgba(52,211,153,0.1)',
                }}
              >
                <Bird size={56} className="text-emerald-400/40 mb-2" />
                <span
                  className="text-[10px] px-2 py-0.5 rounded-lg"
                  style={{
                    background: 'rgba(0,0,0,0.45)',
                    color: '#475569',
                    border: '1px solid rgba(255,255,255,0.05)',
                  }}
                >
                  No image available
                </span>
              </motion.div>

              {/* Names */}
              <motion.div
                className="relative z-10 text-center"
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.45 }}
              >
                <h3
                  className="text-2xl font-bold text-white leading-tight"
                  style={{ fontFamily: 'Outfit, sans-serif' }}
                >
                  {result.name}
                </h3>
                <p className="text-slate-500 text-sm italic mt-1">{result.scientificName}</p>
              </motion.div>
            </div>

            {/* ── RIGHT: Details ── */}
            <div className="p-8 flex flex-col gap-6">
              {/* Confidence */}
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.35 }}
              >
                <p className="text-[10px] font-semibold text-slate-600 uppercase tracking-widest mb-3">
                  Confidence Score
                </p>
                <div className="flex items-center gap-5">
                  <CircularProgress value={result.confidence} size={88} stroke={7} />
                  <div>
                    <p className="text-4xl font-extrabold text-white tabular-nums">
                      {result.confidence.toFixed(1)}
                      <span className="text-2xl text-slate-500">%</span>
                    </p>
                    <p className="text-slate-600 text-xs mt-1">Model certainty · Top-1</p>
                  </div>
                </div>

                {/* Confidence bar */}
                <div
                  className="mt-4 h-1.5 rounded-full overflow-hidden"
                  style={{ background: 'rgba(255,255,255,0.05)' }}
                >
                  <motion.div
                    className="h-full rounded-full"
                    style={{
                      background: 'linear-gradient(90deg, #22d3ee, #818cf8)',
                      boxShadow: '0 0 8px rgba(34,211,238,0.5)',
                    }}
                    initial={{ width: '0%' }}
                    animate={{ width: `${result.confidence}%` }}
                    transition={{ duration: 1.4, ease: [0.16, 1, 0.3, 1], delay: 0.4 }}
                  />
                </div>
              </motion.div>

              <Divider />

              {/* Description */}
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.45 }}
              >
                <p className="text-[10px] font-semibold text-slate-600 uppercase tracking-widest mb-2">
                  About this Species
                </p>
                <p className="text-slate-300 text-sm leading-relaxed">{result.description}</p>
              </motion.div>

              <Divider />

              {/* Meta tags */}
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.55 }}
                className="grid grid-cols-2 gap-3"
              >
                <MetaChip
                  icon={<Leaf size={12} className="text-emerald-400" />}
                  label="Habitat"
                  value={result.habitat}
                />
                <MetaChip
                  icon={<MapPin size={12} className="text-cyan-400" />}
                  label="Region"
                  value={result.region}
                />
              </motion.div>
            </div>
          </div>
        </motion.div>

        {/* ── Bottom disclaimer + reset ── */}
        <motion.div
          variants={itemVariants}
          className="flex flex-col sm:flex-row items-center justify-between gap-4 mt-6 px-2"
        >
          <p className="flex items-center gap-1.5 text-slate-700 text-[11px]">
            <Info size={11} className="flex-shrink-0" />
            Closed-set classification — only trained species can be predicted.
          </p>

          <motion.button
            id="identify-another-btn"
            whileHover={{ scale: 1.04 }}
            whileTap={{ scale: 0.96 }}
            onClick={onReset}
            className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-semibold text-white transition-all"
            style={{
              background: 'rgba(255,255,255,0.05)',
              border: '1px solid rgba(255,255,255,0.1)',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = 'rgba(255,255,255,0.09)'
              e.currentTarget.style.border = '1px solid rgba(255,255,255,0.18)'
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = 'rgba(255,255,255,0.05)'
              e.currentTarget.style.border = '1px solid rgba(255,255,255,0.1)'
            }}
          >
            <RotateCcw size={14} />
            Identify Another Bird
          </motion.button>
        </motion.div>
      </motion.div>
    </section>
  )
}

/* ── Tiny helpers ─────────────────────────────────────────── */
function Divider() {
  return <div style={{ borderTop: '1px solid rgba(255,255,255,0.04)' }} />
}

function MetaChip({ icon, label, value }) {
  return (
    <div
      className="p-3.5 rounded-xl"
      style={{
        background: 'rgba(255,255,255,0.03)',
        border: '1px solid rgba(255,255,255,0.05)',
      }}
    >
      <div className="flex items-center gap-1.5 mb-1">
        {icon}
        <p className="text-[10px] text-slate-600 font-semibold uppercase tracking-wider">{label}</p>
      </div>
      <p className="text-sm text-slate-300 font-medium leading-snug">{value}</p>
    </div>
  )
}
