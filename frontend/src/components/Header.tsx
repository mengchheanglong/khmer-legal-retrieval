import React from 'react'
import {
  Menu,
  Sun,
  Moon,
  Plus,
  Scale,
  Sparkles,
  SlidersHorizontal,
} from 'lucide-react'
import type { ModelOption } from '../types'

interface HeaderProps {
  onToggleSidebar: () => void
  onNewChat: () => void
  darkMode: boolean
  onToggleDarkMode: () => void
  activeModel: ModelOption
  topK: number
  setTopK: (k: number) => void
}

export const Header: React.FC<HeaderProps> = ({
  onToggleSidebar,
  onNewChat,
  darkMode,
  onToggleDarkMode,
  activeModel,
  topK,
  setTopK,
}) => {
  return (
    <header className="h-14 border-b border-border/80 bg-background/80 backdrop-blur-md px-4 flex items-center justify-between shrink-0 sticky top-0 z-20">
      {/* Left: Mobile Menu Toggle & Brand */}
      <div className="flex items-center gap-3">
        <button
          onClick={onToggleSidebar}
          className="p-2 rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted transition-colors md:hidden cursor-pointer"
          aria-label="Toggle Navigation Sidebar"
        >
          <Menu className="w-5 h-5" />
        </button>

        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-linear-to-tr from-blue-600 to-indigo-600 text-white flex items-center justify-center font-bold text-xs shadow-xs">
            <Scale className="w-4 h-4" />
          </div>
          <div>
            <div className="flex items-center gap-1.5">
              <span className="font-bold text-sm tracking-tight text-foreground">
                Khmer Legal AI
              </span>
              <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20 hidden sm:inline-flex items-center gap-1">
                <Sparkles className="w-2.5 h-2.5" /> 51.4% Recall@5
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Center/Right Controls */}
      <div className="flex items-center gap-2">
        {/* Active Model Pill */}
        <div className="hidden lg:flex items-center gap-2 px-3 py-1 rounded-full border border-border/70 bg-card text-xs font-medium text-foreground">
          <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
          <span>{activeModel.name}</span>
          <span className="text-[10px] font-mono text-muted-foreground">({activeModel.params})</span>
        </div>

        {/* Top-K Selector */}
        <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-lg border border-border/70 bg-card text-xs text-muted-foreground">
          <SlidersHorizontal className="w-3.5 h-3.5" />
          <span>Top-k:</span>
          <select
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
            className="bg-transparent text-foreground font-semibold outline-none cursor-pointer"
          >
            <option value={3} className="bg-card text-foreground">3</option>
            <option value={5} className="bg-card text-foreground">5</option>
            <option value={10} className="bg-card text-foreground">10</option>
          </select>
        </div>

        {/* New Chat Button */}
        <button
          onClick={onNewChat}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-primary text-primary-foreground hover:bg-primary/90 transition-all shadow-xs cursor-pointer"
        >
          <Plus className="w-3.5 h-3.5" />
          <span className="hidden sm:inline">New Chat</span>
        </button>

        {/* Dark/Light Theme Toggle */}
        <button
          onClick={onToggleDarkMode}
          className="p-2 rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted transition-colors cursor-pointer"
          title={darkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
        >
          {darkMode ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
        </button>
      </div>
    </header>
  )
}
