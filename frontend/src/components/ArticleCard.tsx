import React, { useState } from 'react'
import { Copy, Check, Layers } from 'lucide-react'
import type { SourceArticle } from '../types'

interface ArticleCardProps {
  article: SourceArticle
  index?: number
}

export const ArticleCard: React.FC<ArticleCardProps> = ({ article, index }) => {
  const [copied, setCopied] = useState(false)

  const isCriminal = article.law_name.toLowerCase().includes('criminal')
  const lawLabel = isCriminal ? 'Criminal Code 2009' : 'Civil Code 2007'

  const handleCopy = () => {
    navigator.clipboard.writeText(`មាត្រា ${article.article_number} (${article.law_name}):\n${article.content}`)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const hierarchy = [article.book, article.chapter, article.section].filter(Boolean)

  let scoreLabel = ''
  if (article.dense_score !== undefined) {
    scoreLabel = `Dense Sim: ${article.dense_score.toFixed(4)}`
  } else if (article.sparse_score !== undefined) {
    scoreLabel = `BM25: ${article.sparse_score.toFixed(2)}`
  } else if (article.relevance_score !== undefined) {
    scoreLabel = `Score: ${article.relevance_score.toFixed(4)}`
  }

  return (
    <div
      className={`rounded-xl border p-4 transition-all duration-200 bg-card/60 hover:bg-card/90 shadow-xs ${
        isCriminal
          ? 'border-l-4 border-l-red-500 border-border/70'
          : 'border-l-4 border-l-blue-500 border-border/70'
      }`}
    >
      {/* Top Header */}
      <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
        <div className="flex items-center gap-2">
          <span
            className={`text-xs font-semibold px-2.5 py-0.5 rounded-full border ${
              isCriminal
                ? 'bg-red-500/10 text-red-600 dark:text-red-400 border-red-500/20'
                : 'bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/20'
            }`}
          >
            {lawLabel}
          </span>
          <h4 className="text-sm font-bold tracking-tight text-foreground font-khmer">
            {index ? `#${index}. ` : ''}មាត្រា {article.article_number}
            {article.article_title && ` — ${article.article_title}`}
          </h4>
        </div>

        <div className="flex items-center gap-2">
          {scoreLabel && (
            <span className="text-[11px] font-mono font-medium px-2 py-0.5 rounded-md bg-muted text-muted-foreground border border-border/50">
              {scoreLabel}
            </span>
          )}
          <button
            onClick={handleCopy}
            className="p-1 rounded-md text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
            title="Copy statutory text"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-green-500" /> : <Copy className="w-3.5 h-3.5" />}
          </button>
        </div>
      </div>

      {/* Hierarchy Breadcrumbs */}
      {hierarchy.length > 0 && (
        <div className="flex items-center gap-1 text-[11px] text-muted-foreground mb-3 font-khmer flex-wrap">
          <Layers className="w-3 h-3 shrink-0 opacity-70" />
          <span>{hierarchy.join(' › ')}</span>
        </div>
      )}

      {/* Article Content in Khmer Font */}
      <div className="font-khmer text-[13.5px] leading-relaxed text-foreground/90 bg-muted/30 p-3 rounded-lg border border-border/40 whitespace-pre-wrap">
        {article.content}
      </div>
    </div>
  )
}
