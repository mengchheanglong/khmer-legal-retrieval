import React, { useState, useEffect } from 'react'
import { Sparkles, Loader2, Database, BrainCircuit, Scale } from 'lucide-react'

interface LoadingMessageProps {
  modelName: string
  backendStage?: { stage: string; message: string } | null
}

export const LoadingMessage: React.FC<LoadingMessageProps> = ({ modelName, backendStage }) => {
  const [elapsed, setElapsed] = useState(0)

  useEffect(() => {
    const start = performance.now()
    const timer = setInterval(() => {
      setElapsed((performance.now() - start) / 1000)
    }, 100)
    return () => clearInterval(timer)
  }, [])

  // Dynamic step message based on real backend stage, with graceful local fallback
  let currentStep = {
    icon: Database,
    text: `Searching 1,976 articles via ${modelName}...`,
    stage: 'Passage Retrieval',
  }

  if (backendStage) {
    if (backendStage.stage === 'retrieval') {
      currentStep = {
        icon: Database,
        text: backendStage.message,
        stage: 'Passage Retrieval',
      }
    } else if (backendStage.stage === 'ranking') {
      currentStep = {
        icon: Scale,
        text: backendStage.message,
        stage: 'Context Grounding',
      }
    } else if (backendStage.stage === 'generation') {
      currentStep = {
        icon: BrainCircuit,
        text: backendStage.message,
        stage: 'Legal Synthesis',
      }
    } else if (backendStage.stage === 'verification') {
      currentStep = {
        icon: Sparkles,
        text: backendStage.message,
        stage: 'Citation Verification',
      }
    }
  } else {
    if (elapsed >= 0.8 && elapsed < 2.0) {
      currentStep = {
        icon: Scale,
        text: 'Matching Civil Code (2007) & Criminal Code (2009) provisions...',
        stage: 'Context Grounding',
      }
    } else if (elapsed >= 2.0) {
      currentStep = {
        icon: BrainCircuit,
        text: 'Synthesizing citation-grounded legal response with DeepSeek Flash...',
        stage: 'Legal Synthesis',
      }
    }
  }

  const StepIcon = currentStep.icon

  return (
    <div className="flex justify-start gap-3 max-w-4xl mx-auto px-4 py-4 animate-in fade-in duration-300">
      {/* Animated Bot Avatar */}
      <div className="relative">
        <div className="w-8 h-8 rounded-full bg-linear-to-tr from-blue-600 to-indigo-500 text-white flex items-center justify-center shrink-0 shadow-md ring-4 ring-primary/20 animate-pulse">
          <Sparkles className="w-4 h-4 animate-spin text-white duration-3000" />
        </div>
      </div>

      {/* Loading Content Box */}
      <div className="flex-1 max-w-[92%] md:max-w-[88%] space-y-3">
        {/* Header with live timer */}
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <span className="font-semibold text-foreground">Khmer Legal Assistant</span>
          <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-primary/10 text-primary text-[10px] font-medium border border-primary/20">
            <Loader2 className="w-2.5 h-2.5 animate-spin" />
            <span>Thinking... ({elapsed.toFixed(1)}s)</span>
          </span>
        </div>

        {/* Dynamic Pipeline Progress Card */}
        <div className="rounded-2xl border border-border/80 bg-card/60 backdrop-blur-xs p-4 md:p-5 shadow-xs space-y-3.5">
          {/* Active Pipeline Stage */}
          <div className="flex items-center gap-2.5 text-xs text-foreground font-medium bg-muted/40 p-2.5 rounded-xl border border-border/50">
            <div className="p-1.5 rounded-lg bg-primary/10 text-primary">
              <StepIcon className="w-4 h-4" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center justify-between text-[10px] text-muted-foreground mb-0.5">
                <span className="font-semibold uppercase tracking-wider text-primary">
                  {currentStep.stage}
                </span>
                <span className="font-mono">{elapsed.toFixed(1)}s</span>
              </div>
              <p className="truncate text-xs font-khmer text-foreground/90">
                {currentStep.text}
              </p>
            </div>
          </div>

          {/* Shimmer Skeleton Lines */}
          <div className="space-y-2.5 pt-1">
            <div className="h-3.5 bg-muted/60 rounded-md w-11/12 animate-pulse" />
            <div className="h-3.5 bg-muted/50 rounded-md w-full animate-pulse delay-75" />
            <div className="h-3.5 bg-muted/40 rounded-md w-4/5 animate-pulse delay-150" />
          </div>

          {/* Bouncing dots indicator */}
          <div className="flex items-center gap-1.5 pt-1 text-muted-foreground text-xs">
            <div className="w-1.5 h-1.5 rounded-full bg-primary animate-bounce [animation-delay:-0.3s]" />
            <div className="w-1.5 h-1.5 rounded-full bg-primary animate-bounce [animation-delay:-0.15s]" />
            <div className="w-1.5 h-1.5 rounded-full bg-primary animate-bounce" />
            <span className="text-[11px] text-muted-foreground font-khmer ml-1">
              កំពុងស្រាវជ្រាវ និងវិភាគមាត្រាច្បាប់...
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}
