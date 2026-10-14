import React, { useRef, useEffect } from 'react'
import { ArrowUp, Loader2, Filter, X } from 'lucide-react'
import type { StatuteFilter } from '../types'

interface ChatInputProps {
  input: string
  setInput: (value: string) => void
  onSend: (text: string) => void
  isLoading: boolean
  statuteFilter: StatuteFilter
  setStatuteFilter: (filter: StatuteFilter) => void
}

export const ChatInput: React.FC<ChatInputProps> = ({
  input,
  setInput,
  onSend,
  isLoading,
  statuteFilter,
  setStatuteFilter,
}) => {
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 160)}px`
    }
  }, [input])

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault()
    if (!input.trim() || isLoading) return
    onSend(input.trim())
    setInput('')
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  return (
    <div className="w-full max-w-4xl mx-auto px-4 pb-4 pt-2">
      <form
        onSubmit={handleSubmit}
        className="relative rounded-2xl border border-border/80 bg-card/90 backdrop-blur-md shadow-lg p-2.5 transition-all focus-within:border-primary/60 focus-within:ring-2 focus-within:ring-primary/10"
      >
        {/* Text Input */}
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a Cambodian legal question in Khmer or English... (e.g. តើកិច្ចសន្យាបង្កើតឡើងដោយរបៀបណា?)"
          rows={1}
          disabled={isLoading}
          className="w-full resize-none bg-transparent px-3 py-2 text-sm md:text-base font-khmer text-foreground placeholder:text-muted-foreground/60 outline-none max-h-40 overflow-y-auto leading-relaxed"
        />

        {/* Bottom Action Bar */}
        <div className="flex items-center justify-between pt-2 border-t border-border/40 px-1 mt-1">
          {/* Statute Filter Pills */}
          <div className="flex items-center gap-1.5 overflow-x-auto text-xs py-0.5">
            <span className="text-[11px] text-muted-foreground mr-1 hidden sm:inline-flex items-center gap-1">
              <Filter className="w-3 h-3" /> Scope:
            </span>
            <button
              type="button"
              onClick={() => setStatuteFilter('ALL')}
              className={`px-2.5 py-1 rounded-full text-[11px] font-medium transition-all ${
                statuteFilter === 'ALL'
                  ? 'bg-primary text-primary-foreground shadow-xs'
                  : 'bg-muted text-muted-foreground hover:bg-muted/80'
              }`}
            >
              All 1,976 Articles
            </button>
            <button
              type="button"
              onClick={() => setStatuteFilter('CIVIL')}
              className={`px-2.5 py-1 rounded-full text-[11px] font-medium transition-all ${
                statuteFilter === 'CIVIL'
                  ? 'bg-blue-600 text-white shadow-xs'
                  : 'bg-muted text-muted-foreground hover:bg-muted/80'
              }`}
            >
              Civil Code (1,304)
            </button>
            <button
              type="button"
              onClick={() => setStatuteFilter('CRIMINAL')}
              className={`px-2.5 py-1 rounded-full text-[11px] font-medium transition-all ${
                statuteFilter === 'CRIMINAL'
                  ? 'bg-red-600 text-white shadow-xs'
                  : 'bg-muted text-muted-foreground hover:bg-muted/80'
              }`}
            >
              Criminal Code (672)
            </button>
          </div>

          {/* Right Action Buttons */}
          <div className="flex items-center gap-1.5 ml-2">
            {input.trim() && !isLoading && (
              <button
                type="button"
                onClick={() => setInput('')}
                className="p-1.5 text-muted-foreground hover:text-foreground rounded-full hover:bg-muted transition-colors"
                title="Clear input"
              >
                <X className="w-4 h-4" />
              </button>
            )}

            <button
              type="submit"
              disabled={!input.trim() || isLoading}
              className={`p-2 rounded-xl transition-all ${
                input.trim() && !isLoading
                  ? 'bg-primary text-primary-foreground hover:bg-primary/90 shadow-sm cursor-pointer'
                  : 'bg-muted text-muted-foreground/40 cursor-not-allowed'
              }`}
              title="Send legal question (Enter)"
            >
              {isLoading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <ArrowUp className="w-4 h-4" />
              )}
            </button>
          </div>
        </div>
      </form>

      {/* Legal Prototype Disclaimer */}
      <div className="text-center mt-2">
        <p className="text-[11px] text-muted-foreground/75 leading-tight">
          Academic DL Legal Assistant · Grounded in official Civil & Criminal Code gazettes (ODC CC-BY-SA-4.0).
          Not formal legal counsel.
        </p>
      </div>
    </div>
  )
}
