import React, { useState, useEffect } from 'react'
import { Copy, Check, Layers, FileText } from 'lucide-react'
import type { SourceArticle } from '../types'

interface ArticleCardProps {
  article: SourceArticle
  index?: number
}

export const ArticleCard: React.FC<ArticleCardProps> = ({ article, index }) => {
  const [copied, setCopied] = useState(false)
  const [contentText, setContentText] = useState<string>(
    article.content || (article as any).content_preview || ''
  )
  const [hierarchyData, setHierarchyData] = useState({
    title: article.article_title,
    book: article.book,
    chapter: article.chapter,
    section: article.section,
  })

  const isCriminal = article.law_name.toLowerCase().includes('criminal')
  const lawLabel = isCriminal ? 'Criminal Code 2009' : 'Civil Code 2007'

  // Auto-hydrate full article content if loaded from legacy cache with only preview (<= 200 chars)
  useEffect(() => {
    if (!article.content || article.content.length <= 200) {
      const fetchFullArticle = async () => {
        try {
          const res = await fetch(
            `/api/v1/articles/${encodeURIComponent(article.law_name)}/${article.article_number}`
          )
          if (res.ok) {
            const data = await res.json()
            if (data.content && data.content.length > contentText.length) {
              setContentText(data.content)
              setHierarchyData({
                title: data.article_title || article.article_title,
                book: data.book || article.book,
                chapter: data.chapter || article.chapter,
                section: data.section || article.section,
              })
            }
          }
        } catch {
          // Keep current text fallback
        }
      }
      fetchFullArticle()
    } else {
      setContentText(article.content)
    }
  }, [article.article_number, article.law_name, article.content])

  const handleCopy = () => {
    navigator.clipboard.writeText(`មាត្រា ${article.article_number} (${lawLabel}):\n${contentText}`)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const hierarchy = [hierarchyData.book, hierarchyData.chapter, hierarchyData.section].filter(Boolean)

  let scoreLabel = ''
  if (article.dense_score !== undefined && article.dense_score !== null) {
    scoreLabel = `Dense Sim: ${article.dense_score.toFixed(4)}`
  } else if (article.sparse_score !== undefined && article.sparse_score !== null) {
    scoreLabel = `BM25: ${article.sparse_score.toFixed(2)}`
  } else if (article.relevance_score !== undefined && article.relevance_score !== null) {
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
            {hierarchyData.title && ` — ${hierarchyData.title}`}
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
            className="p-1 rounded-md text-muted-foreground hover:text-foreground hover:bg-muted transition-colors cursor-pointer"
            title="Copy complete statutory text"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-green-500" /> : <Copy className="w-3.5 h-3.5" />}
          </button>
        </div>
      </div>

      {/* Hierarchy Breadcrumbs & Char Count */}
      <div className="flex items-center justify-between gap-2 mb-3 text-xs text-muted-foreground font-khmer flex-wrap">
        {hierarchy.length > 0 && (
          <div className="flex items-center gap-1.5 flex-wrap">
            <Layers className="w-3.5 h-3.5 shrink-0 opacity-70" />
            <span>{hierarchy.join(' › ')}</span>
          </div>
        )}
        {contentText.length > 0 && (
          <span className="inline-flex items-center gap-1 text-[10px] font-mono text-muted-foreground/75 bg-muted/60 px-2 py-0.5 rounded">
            <FileText className="w-3 h-3 opacity-60" />
            {contentText.length} chars (Full Article)
          </span>
        )}
      </div>

      {/* Article Content in Khmer Font */}
      <div className="font-khmer text-sm md:text-[14.5px] leading-relaxed text-foreground/95 bg-muted/40 p-3.5 md:p-4 rounded-xl border border-border/50 whitespace-pre-wrap select-text">
        {contentText || 'គ្មានខ្លឹមសារបង្ហាញ'}
      </div>
    </div>
  )
}
