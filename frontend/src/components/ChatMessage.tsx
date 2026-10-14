import React, { useState } from 'react'
import {
  User,
  Copy,
  Check,
  ShieldCheck,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  BookOpen,
  Sparkles,
} from 'lucide-react'
import type { Message } from '../types'
import { ArticleCard } from './ArticleCard'

interface ChatMessageProps {
  message: Message
}

export const ChatMessage: React.FC<ChatMessageProps> = ({ message }) => {
  const [copied, setCopied] = useState(false)
  const [showSources, setShowSources] = useState(true)

  const isUser = message.role === 'user'

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  if (isUser) {
    return (
      <div className="flex justify-end gap-3 max-w-4xl mx-auto px-4 py-3">
        <div className="flex flex-col items-end max-w-[85%] md:max-w-[75%]">
          <div className="bg-primary text-primary-foreground px-4 py-3 rounded-2xl rounded-tr-xs shadow-sm text-sm font-khmer leading-relaxed whitespace-pre-wrap">
            {message.content}
          </div>
          <span className="text-[10px] text-muted-foreground mt-1 px-1">
            {message.timestamp}
          </span>
        </div>
        <div className="w-8 h-8 rounded-full bg-primary/20 text-primary flex items-center justify-center shrink-0 border border-primary/30 mt-0.5">
          <User className="w-4 h-4" />
        </div>
      </div>
    )
  }

  return (
    <div className="flex justify-start gap-3 max-w-4xl mx-auto px-4 py-4 group">
      {/* Bot Avatar */}
      <div className="w-8 h-8 rounded-full bg-linear-to-tr from-blue-600 to-indigo-500 text-white flex items-center justify-center shrink-0 shadow-sm mt-0.5 ring-2 ring-primary/20">
        <Sparkles className="w-4 h-4" />
      </div>

      {/* Assistant Content Box */}
      <div className="flex-1 max-w-[92%] md:max-w-[88%] space-y-3">
        {/* Header with model badge & copy */}
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-foreground">Khmer Legal Assistant</span>
            {message.modelUsed && (
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-muted font-medium border border-border/50">
                {message.modelUsed}
              </span>
            )}
            {message.retrievalMs && (
              <span className="text-[10px] text-muted-foreground">
                · {message.retrievalMs.toFixed(0)}ms
              </span>
            )}
          </div>
          <button
            onClick={handleCopy}
            className="flex items-center gap-1 text-[11px] hover:text-foreground p-1 rounded-md hover:bg-muted transition-colors opacity-70 group-hover:opacity-100"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-green-500" /> : <Copy className="w-3.5 h-3.5" />}
            <span>{copied ? 'Copied' : 'Copy'}</span>
          </button>
        </div>

        {/* Formatted Answer */}
        <div className="font-khmer text-[14px] leading-relaxed text-foreground bg-card/40 border border-border/60 rounded-2xl p-4 md:p-5 shadow-xs whitespace-pre-wrap">
          {message.content}
        </div>

        {/* Cited Authorities Verification */}
        {message.citations && message.citations.length > 0 && (
          <div className="rounded-xl border border-border/70 bg-muted/20 p-3 space-y-2">
            <div className="flex items-center gap-1.5 text-xs font-semibold text-foreground">
              <ShieldCheck className="w-4 h-4 text-emerald-500" />
              <span>Cited Statutory Authorities ({message.citations.length})</span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 pt-1">
              {message.citations.map((cit, idx) => (
                <div
                  key={idx}
                  className="flex items-center justify-between p-2 rounded-lg bg-card/80 border border-border/50 text-xs"
                >
                  <span className="font-khmer font-medium text-foreground truncate pr-2">
                    {cit.law_name} · មាត្រា {cit.article_number}
                  </span>
                  {cit.is_verified ? (
                    <span className="inline-flex items-center gap-1 text-[10px] font-semibold text-emerald-600 dark:text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded-full border border-emerald-500/20 shrink-0">
                      <ShieldCheck className="w-3 h-3" /> Verified
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1 text-[10px] font-semibold text-amber-600 dark:text-amber-400 bg-amber-500/10 px-2 py-0.5 rounded-full border border-amber-500/20 shrink-0">
                      <AlertTriangle className="w-3 h-3" /> Unverified
                    </span>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Retrieved Context Drawer */}
        {message.sourceArticles && message.sourceArticles.length > 0 && (
          <div className="rounded-xl border border-border/70 overflow-hidden bg-card/30">
            <button
              onClick={() => setShowSources(!showSources)}
              className="w-full flex items-center justify-between p-3 text-xs font-medium text-muted-foreground hover:text-foreground hover:bg-muted/40 transition-colors"
            >
              <div className="flex items-center gap-2">
                <BookOpen className="w-4 h-4 text-primary" />
                <span className="font-semibold text-foreground">
                  Retrieved Statutory Context ({message.sourceArticles.length} Articles)
                </span>
                <span className="text-[10px] bg-primary/10 text-primary px-2 py-0.5 rounded-full font-medium">
                  Ground Truth
                </span>
              </div>
              {showSources ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </button>

            {showSources && (
              <div className="p-3 pt-1 space-y-2.5 border-t border-border/40">
                {message.sourceArticles.map((art, idx) => (
                  <ArticleCard key={idx} article={art} index={idx + 1} />
                ))}
              </div>
            )}
          </div>
        )}

        <div className="text-[10px] text-muted-foreground px-1">
          {message.timestamp} · Grounded in official Civil & Criminal Codes
        </div>
      </div>
    </div>
  )
}
