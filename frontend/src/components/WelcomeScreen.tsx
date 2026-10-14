import React from 'react'
import { Sparkles, Scale, BookMarked, ShieldAlert, FileText, ArrowRight } from 'lucide-react'

interface WelcomeScreenProps {
  onSelectPrompt: (prompt: string) => void
}

const STARTER_PROMPTS = [
  {
    icon: FileText,
    law: 'Civil Code 2007',
    badgeClass: 'text-blue-500 bg-blue-500/10 border-blue-500/20',
    title: 'Formation of Legal Contracts (Art. 311 & 336)',
    prompt: 'តើកិច្ចសន្យាបង្កើតឡើងដោយរបៀបណា យោងតាមក្រមរដ្ឋប្បវេណី?',
    enSummary: 'Requirements and validity of contractual offer and acceptance',
  },
  {
    icon: Scale,
    law: 'Criminal Code 2009',
    badgeClass: 'text-red-500 bg-red-500/10 border-red-500/20',
    title: 'Legitimate Self-Defense (Art. 33)',
    prompt: 'តើអ្វីទៅជាការការពារស្របច្បាប់ (Self-Defense) ក្នុងក្រមព្រហ្មទណ្ឌ?',
    enSummary: 'Conditions exempting criminal liability under self-defense',
  },
  {
    icon: BookMarked,
    law: 'Civil Code 2007',
    badgeClass: 'text-blue-500 bg-blue-500/10 border-blue-500/20',
    title: "Seller's Liability for Hidden Defects (Art. 544)",
    prompt: 'តើអ្នកលក់មានការទទួលខុសត្រូវលើវិការៈនៃវត្ថុទិញលក់យ៉ាងដូចម្តេច?',
    enSummary: 'Buyer remedies for latent defects in sale agreements',
  },
  {
    icon: ShieldAlert,
    law: 'Criminal Code 2009',
    badgeClass: 'text-red-500 bg-red-500/10 border-red-500/20',
    title: 'Classification of Criminal Offenses (Art. 46)',
    prompt: 'តើបទល្មើសព្រហ្មទណ្ឌត្រូវបានបែងចែកជាប៉ុន្មានថ្នាក់?',
    enSummary: 'Felonies, misdemeanors, and petty offenses distinctions',
  },
]

export const WelcomeScreen: React.FC<WelcomeScreenProps> = ({ onSelectPrompt }) => {
  return (
    <div className="flex-1 flex flex-col items-center justify-center max-w-3xl mx-auto px-4 py-8 text-center">
      {/* Top Hero Icon */}
      <div className="relative mb-5">
        <div className="w-16 h-16 rounded-2xl bg-linear-to-tr from-blue-600 to-indigo-600 text-white flex items-center justify-center shadow-lg ring-4 ring-primary/10">
          <Scale className="w-8 h-8" />
        </div>
        <div className="absolute -bottom-1 -right-1 w-6 h-6 rounded-full bg-emerald-500 text-white flex items-center justify-center ring-2 ring-background">
          <Sparkles className="w-3.5 h-3.5" />
        </div>
      </div>

      {/* Category Pill */}
      <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-primary/10 text-primary border border-primary/20 mb-3">
        <span>🇰🇭</span>
        <span>Khmer Legal NLP · 1,976 Articles</span>
      </div>

      {/* Main Title & Subtitle */}
      <h1 className="text-2xl md:text-3xl font-extrabold tracking-tight text-foreground mb-2">
        Khmer Legal AI Assistant
      </h1>
      <p className="text-sm md:text-base text-muted-foreground max-w-xl mb-8 leading-relaxed font-sans">
        Neural passage retrieval and citation-verified analysis grounded strictly in the
        official <strong>Civil Code (2007)</strong> and <strong>Criminal Code (2009)</strong>.
      </p>

      {/* Prompt Starter Cards Grid */}
      <div className="w-full grid grid-cols-1 md:grid-cols-2 gap-3 text-left">
        {STARTER_PROMPTS.map((item, idx) => {
          const Icon = item.icon
          return (
            <button
              key={idx}
              onClick={() => onSelectPrompt(item.prompt)}
              className="group relative flex flex-col justify-between p-4 rounded-xl border border-border/70 bg-card/60 hover:bg-card hover:border-primary/50 hover:shadow-md transition-all duration-200 text-left cursor-pointer"
            >
              <div>
                <div className="flex items-center justify-between gap-2 mb-2">
                  <div className="flex items-center gap-2">
                    <div className="p-1.5 rounded-lg bg-muted text-foreground">
                      <Icon className="w-4 h-4 text-primary" />
                    </div>
                    <span className={`text-[11px] font-semibold px-2 py-0.5 rounded-full border ${item.badgeClass}`}>
                      {item.law}
                    </span>
                  </div>
                  <ArrowRight className="w-4 h-4 text-muted-foreground opacity-0 group-hover:opacity-100 group-hover:translate-x-0.5 transition-all" />
                </div>
                <h3 className="text-sm font-semibold text-foreground mb-1 group-hover:text-primary transition-colors">
                  {item.title}
                </h3>
                <p className="font-khmer text-xs text-muted-foreground line-clamp-2 leading-relaxed">
                  {item.prompt}
                </p>
              </div>
              <div className="mt-3 pt-2 border-t border-border/40 text-[11px] text-muted-foreground/80">
                {item.enSummary}
              </div>
            </button>
          )
        })}
      </div>
    </div>
  )
}
