import React, { useState } from 'react'
import {
  Plus,
  MessageSquare,
  Trash2,
  Cpu,
  Database,
  ExternalLink,
  Key,
  X,
  Search,
} from 'lucide-react'
import type { ChatSession, ModelOption } from '../types'

interface SidebarProps {
  isOpen: boolean
  onClose: () => void
  sessions: ChatSession[]
  currentSessionId: string
  onSelectSession: (id: string) => void
  onNewChat: () => void
  onDeleteSession: (id: string) => void
  models: ModelOption[]
  activeModel: ModelOption
  onSelectModel: (model: ModelOption) => void
  apiKey: string
  setApiKey: (key: string) => void
}

export const Sidebar: React.FC<SidebarProps> = ({
  isOpen,
  onClose,
  sessions,
  currentSessionId,
  onSelectSession,
  onNewChat,
  onDeleteSession,
  models,
  activeModel,
  onSelectModel,
  apiKey,
  setApiKey,
}) => {
  const [searchHistory, setSearchHistory] = useState('')
  const [showKeyModal, setShowKeyModal] = useState(false)
  const [tempKey, setTempKey] = useState(apiKey)

  const filteredSessions = sessions.filter((s) =>
    s.title.toLowerCase().includes(searchHistory.toLowerCase())
  )

  const handleSaveKey = () => {
    setApiKey(tempKey)
    setShowKeyModal(false)
  }

  return (
    <>
      {/* Mobile Backdrop */}
      {isOpen && (
        <div
          onClick={onClose}
          className="fixed inset-0 bg-black/50 z-30 md:hidden backdrop-blur-xs transition-opacity"
        />
      )}

      {/* Sidebar Container */}
      <aside
        className={`fixed md:static inset-y-0 left-0 z-40 w-72 md:w-80 bg-sidebar border-r border-border/80 flex flex-col transition-transform duration-200 ease-in-out ${
          isOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'
        }`}
      >
        {/* Top Header & New Chat */}
        <div className="p-3.5 border-b border-border/70 space-y-2.5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground">
              Conversations
            </span>
            <button
              onClick={onClose}
              className="p-1 rounded-md text-muted-foreground hover:text-foreground md:hidden"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          <button
            onClick={() => {
              onNewChat()
              onClose()
            }}
            className="w-full flex items-center justify-center gap-2 px-3 py-2.5 rounded-xl text-xs font-semibold bg-primary text-primary-foreground hover:bg-primary/90 transition-all shadow-xs cursor-pointer"
          >
            <Plus className="w-4 h-4" />
            <span>New Legal Consultation</span>
          </button>

          {/* Search History */}
          <div className="relative">
            <Search className="w-3.5 h-3.5 absolute left-2.5 top-1/2 -translate-y-1/2 text-muted-foreground pointer-events-none" />
            <input
              type="text"
              value={searchHistory}
              onChange={(e) => setSearchHistory(e.target.value)}
              placeholder="Search chat history..."
              className="w-full pl-8 pr-3 py-1.5 text-xs bg-muted/60 border border-border/50 rounded-lg text-foreground placeholder:text-muted-foreground outline-none focus:border-primary/50"
            />
          </div>
        </div>

        {/* Chat History List */}
        <div className="flex-1 overflow-y-auto p-2 space-y-1">
          {filteredSessions.length === 0 ? (
            <div className="p-4 text-center text-xs text-muted-foreground">
              {searchHistory ? 'No matching conversations' : 'No previous conversations yet'}
            </div>
          ) : (
            filteredSessions.map((session) => (
              <div
                key={session.id}
                className={`group flex items-center justify-between px-3 py-2 rounded-lg text-xs transition-colors cursor-pointer ${
                  session.id === currentSessionId
                    ? 'bg-primary/10 text-primary font-semibold'
                    : 'text-muted-foreground hover:bg-muted/70 hover:text-foreground'
                }`}
                onClick={() => {
                  onSelectSession(session.id)
                  onClose()
                }}
              >
                <div className="flex items-center gap-2 truncate pr-2">
                  <MessageSquare className="w-3.5 h-3.5 shrink-0 opacity-70" />
                  <span className="truncate font-khmer">{session.title}</span>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    onDeleteSession(session.id)
                  }}
                  className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-destructive/10 hover:text-destructive transition-all"
                  title="Delete chat"
                >
                  <Trash2 className="w-3 h-3" />
                </button>
              </div>
            ))
          )}
        </div>

        {/* Retrieval Model Selector */}
        <div className="p-3 border-t border-border/70 space-y-2 bg-muted/20">
          <div className="flex items-center justify-between text-xs text-muted-foreground font-semibold">
            <div className="flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-primary" />
              <span>Retriever Architecture</span>
            </div>
            {activeModel.winner && (
              <span className="text-[10px] font-bold text-emerald-600 dark:text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded">
                Winner
              </span>
            )}
          </div>

          <select
            value={activeModel.id}
            onChange={(e) => {
              const selected = models.find((m) => m.id === e.target.value)
              if (selected) onSelectModel(selected)
            }}
            className="w-full text-xs bg-background border border-border/80 rounded-lg p-2 text-foreground font-medium outline-none focus:border-primary cursor-pointer"
          >
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name} ({m.params}) {m.winner ? '🏆' : ''}
              </option>
            ))}
          </select>

          <p className="text-[11px] text-muted-foreground leading-tight">
            {activeModel.type} · Trainable params: {activeModel.params}
          </p>
        </div>

        {/* Corpus Statistics Box */}
        <div className="p-3 border-t border-border/70 bg-card/40 space-y-2 text-xs">
          <div className="flex items-center justify-between text-muted-foreground">
            <span className="flex items-center gap-1.5 font-medium">
              <Database className="w-3.5 h-3.5 text-blue-500" />
              <span>Corpus Size:</span>
            </span>
            <strong className="text-foreground">1,976 Articles</strong>
          </div>
          <div className="grid grid-cols-2 gap-1.5 text-[11px]">
            <div className="p-1.5 rounded-md bg-blue-500/10 text-blue-600 dark:text-blue-400 border border-blue-500/20 text-center font-medium">
              Civil: 1,304 Arts
            </div>
            <div className="p-1.5 rounded-md bg-red-500/10 text-red-600 dark:text-red-400 border border-red-500/20 text-center font-medium">
              Criminal: 672 Arts
            </div>
          </div>
        </div>

        {/* Footer Actions */}
        <div className="p-3 border-t border-border/70 flex items-center justify-between text-xs text-muted-foreground">
          <button
            onClick={() => setShowKeyModal(true)}
            className="flex items-center gap-1 hover:text-foreground p-1 rounded transition-colors"
          >
            <Key className="w-3.5 h-3.5 text-amber-500" />
            <span>{apiKey ? 'API Key Set' : 'Add API Key'}</span>
          </button>

          <a
            href="http://localhost:8501"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1 hover:text-primary p-1 rounded transition-colors"
            title="Open Streamlit Evaluation Dashboard"
          >
            <span>Research UI</span>
            <ExternalLink className="w-3 h-3" />
          </a>
        </div>
      </aside>

      {/* API Key Modal */}
      {showKeyModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-xs">
          <div className="bg-card border border-border rounded-2xl max-w-md w-full p-5 shadow-2xl space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Key className="w-5 h-5 text-amber-500" />
                <h3 className="font-bold text-base text-foreground">DeepSeek API Key</h3>
              </div>
              <button
                onClick={() => setShowKeyModal(false)}
                className="p-1 rounded-md text-muted-foreground hover:text-foreground"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <p className="text-xs text-muted-foreground leading-relaxed">
              If your backend already has <code>DEEPSEEK_API_KEY</code> set in <code>.env</code>,
              you do not need to enter one here. Enter an override key if you wish to query directly.
            </p>

            <input
              type="password"
              value={tempKey}
              onChange={(e) => setTempKey(e.target.value)}
              placeholder="sk-..."
              className="w-full text-xs bg-muted/50 border border-border rounded-lg p-2.5 text-foreground placeholder:text-muted-foreground outline-none focus:border-primary"
            />

            <div className="flex justify-end gap-2 pt-2">
              <button
                onClick={() => setShowKeyModal(false)}
                className="px-3 py-1.5 rounded-lg text-xs font-medium text-muted-foreground hover:text-foreground hover:bg-muted"
              >
                Cancel
              </button>
              <button
                onClick={handleSaveKey}
                className="px-4 py-1.5 rounded-lg text-xs font-semibold bg-primary text-primary-foreground hover:bg-primary/90"
              >
                Save Key
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
