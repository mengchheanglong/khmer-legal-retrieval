export interface SourceArticle {
  law_name: string
  article_number: number
  article_title?: string
  book?: string
  chapter?: string
  section?: string
  content: string
  relevance_score?: number
  dense_score?: number
  sparse_score?: number
}

export interface Citation {
  law_name: string
  article_number: number
  is_verified: boolean
  excerpt?: string
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: string
  citations?: Citation[]
  sourceArticles?: SourceArticle[]
  retrievalMs?: number
  modelUsed?: string
}

export interface ChatSession {
  id: string
  title: string
  messages: Message[]
  createdAt: string
  updatedAt: string
}

export type StatuteFilter = 'ALL' | 'CIVIL' | 'CRIMINAL'

export interface ModelOption {
  id: string
  name: string
  tag: string
  type: string
  params: string
  winner?: boolean
}
