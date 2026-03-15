import { useCallback, useEffect, useRef, useState } from 'react'
import type { ServerMessage, Recipe, RecipeGraphData } from '../types/messages'

interface UseWebSocketReturn {
  connected: boolean
  sendMessage: (content: string) => void
  messages: ServerMessage[]
  activeAgent: string | null
  recipe: Recipe | null
  graph: RecipeGraphData | null
}

export function useWebSocket(sessionId: string): UseWebSocketReturn {
  const wsRef = useRef<WebSocket | null>(null)
  const [connected, setConnected] = useState(false)
  const [messages, setMessages] = useState<ServerMessage[]>([])
  const [activeAgent, setActiveAgent] = useState<string | null>(null)
  const [recipe, setRecipe] = useState<Recipe | null>(null)
  const [graph, setGraph] = useState<RecipeGraphData | null>(null)
  const reconnectTimeout = useRef<ReturnType<typeof setTimeout>>()

  const connect = useCallback(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.host
    const ws = new WebSocket(`${protocol}//${host}/ws/${sessionId}`)

    ws.onopen = () => {
      setConnected(true)
    }

    ws.onclose = () => {
      setConnected(false)
      wsRef.current = null
      // Reconnect with backoff
      reconnectTimeout.current = setTimeout(connect, 3000)
    }

    ws.onmessage = (event) => {
      const data: ServerMessage = JSON.parse(event.data)
      setMessages((prev) => [...prev, data])

      switch (data.type) {
        case 'agent_event':
          setActiveAgent(data.status === 'working' ? data.agent : null)
          break
        case 'agent_response':
          setActiveAgent(null)
          break
        case 'recipe_update':
          setRecipe(data.recipe)
          break
        case 'graph_update':
          setGraph(data.graph)
          break
      }
    }

    wsRef.current = ws
  }, [sessionId])

  useEffect(() => {
    connect()
    return () => {
      clearTimeout(reconnectTimeout.current)
      wsRef.current?.close()
    }
  }, [connect])

  const sendMessage = useCallback((content: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'user_message', content }))
    }
  }, [])

  return { connected, sendMessage, messages, activeAgent, recipe, graph }
}
