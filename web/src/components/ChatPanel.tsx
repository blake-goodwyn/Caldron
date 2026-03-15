import { useEffect, useRef, useState } from 'react'
import type { ChatMessage } from '../types/messages'
import { MessageBubble } from './MessageBubble'
import { AgentIndicator } from './AgentIndicator'

interface ChatPanelProps {
  messages: ChatMessage[]
  activeAgent: string | null
  connected: boolean
  onSend: (content: string) => void
}

export function ChatPanel({ messages, activeAgent, connected, onSend }: ChatPanelProps) {
  const [input, setInput] = useState('')
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages, activeAgent])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const trimmed = input.trim()
    if (!trimmed) return
    onSend(trimmed)
    setInput('')
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-caldron-mid/20">
        <h2 className="text-caldron-cream font-semibold text-lg">Caldron</h2>
        <div className="flex items-center gap-2">
          <span className={`h-2 w-2 rounded-full ${connected ? 'bg-green-400' : 'bg-red-400'}`} />
          <span className="text-xs text-caldron-light/50">
            {connected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </div>

      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto chat-scroll p-4 space-y-3">
        {messages.length === 0 && (
          <div className="flex items-center justify-center h-full text-caldron-light/30 text-sm">
            Start a conversation to develop a recipe
          </div>
        )}
        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}
        <AgentIndicator agent={activeAgent} />
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="p-4 border-t border-caldron-mid/20">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Describe a recipe idea..."
            disabled={!connected}
            className="flex-1 bg-caldron-dark/50 border border-caldron-mid/30 rounded-xl px-4 py-3 text-sm text-caldron-cream placeholder-caldron-light/30 focus:outline-none focus:border-caldron-mid transition-colors disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={!connected || !input.trim()}
            className="px-5 py-3 bg-caldron-mid hover:bg-caldron-mid/80 disabled:bg-caldron-mid/30 text-white rounded-xl text-sm font-medium transition-colors"
          >
            Send
          </button>
        </div>
      </form>
    </div>
  )
}
