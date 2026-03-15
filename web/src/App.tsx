import { useCallback, useMemo, useState } from 'react'
import { useWebSocket } from './hooks/useWebSocket'
import { ChatPanel } from './components/ChatPanel'
import { RecipeCard } from './components/RecipeCard'
import { RecipeGraph } from './components/RecipeGraph'
import type { ChatMessage } from './types/messages'

function generateId() {
  return crypto.randomUUID()
}

export default function App() {
  const sessionId = useMemo(() => generateId(), [])
  const { connected, sendMessage, messages, activeAgent, recipe, graph } = useWebSocket(sessionId)
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([])

  const handleSend = useCallback(
    (content: string) => {
      // Add user message to chat
      setChatMessages((prev) => [
        ...prev,
        { id: generateId(), role: 'user', content, timestamp: new Date() },
      ])
      sendMessage(content)
    },
    [sendMessage]
  )

  // Convert agent_response server messages to chat messages
  const allChatMessages = useMemo(() => {
    const agentMessages: ChatMessage[] = messages
      .filter((m) => m.type === 'agent_response')
      .map((m) => ({
        id: generateId(),
        role: 'agent' as const,
        content: m.type === 'agent_response' ? m.content : '',
        timestamp: new Date(),
      }))

    // Interleave: user messages are tracked in state, agent responses from server
    const combined = [...chatMessages]
    // Simple approach: append any new agent responses not yet in chat
    for (const am of agentMessages) {
      if (!combined.some((c) => c.role === 'agent' && c.content === am.content)) {
        combined.push(am)
      }
    }
    return combined
  }, [chatMessages, messages])

  const [activeTab, setActiveTab] = useState<'recipe' | 'graph'>('recipe')

  return (
    <div className="h-screen flex flex-col bg-caldron-dark">
      {/* Top bar */}
      <header className="flex items-center px-6 py-3 border-b border-caldron-mid/20">
        <h1 className="text-caldron-cream font-bold text-xl tracking-wide">
          🍲 Caldron
        </h1>
        <span className="ml-3 text-caldron-light/40 text-xs">AI Recipe Development</span>
      </header>

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left panel: Chat */}
        <div className="w-1/2 border-r border-caldron-mid/20 flex flex-col">
          <ChatPanel
            messages={allChatMessages}
            activeAgent={activeAgent}
            connected={connected}
            onSend={handleSend}
          />
        </div>

        {/* Right panel: Recipe + Graph */}
        <div className="w-1/2 flex flex-col">
          {/* Tabs */}
          <div className="flex border-b border-caldron-mid/20">
            <button
              onClick={() => setActiveTab('recipe')}
              className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
                activeTab === 'recipe'
                  ? 'text-caldron-cream border-b-2 border-caldron-accent'
                  : 'text-caldron-light/40 hover:text-caldron-light/70'
              }`}
            >
              Recipe
            </button>
            <button
              onClick={() => setActiveTab('graph')}
              className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
                activeTab === 'graph'
                  ? 'text-caldron-cream border-b-2 border-caldron-accent'
                  : 'text-caldron-light/40 hover:text-caldron-light/70'
              }`}
            >
              Development Graph
            </button>
          </div>

          {/* Tab content */}
          <div className="flex-1 overflow-hidden">
            {activeTab === 'recipe' ? (
              <RecipeCard recipe={recipe} />
            ) : (
              <RecipeGraph graph={graph} />
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
