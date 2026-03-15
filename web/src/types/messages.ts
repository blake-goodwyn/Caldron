export interface UserMessage {
  type: 'user_message'
  content: string
}

export interface AgentEvent {
  type: 'agent_event'
  agent: string
  status: 'working' | 'done'
  content: string | null
}

export interface AgentResponse {
  type: 'agent_response'
  content: string
}

export interface Ingredient {
  name: string
  quantity: number
  unit: string | null
}

export interface Recipe {
  name: string
  ingredients: Ingredient[]
  instructions: string[]
  tags: string[]
  sources: string[]
}

export interface RecipeUpdate {
  type: 'recipe_update'
  recipe: Recipe | null
}

export interface GraphNode {
  node_id: string
  recipe: Recipe | null
}

export interface GraphEdge {
  source: string
  target: string
}

export interface RecipeGraphData {
  foundational_recipe_node: string | null
  nodes: GraphNode[]
  edges: GraphEdge[]
}

export interface GraphUpdate {
  type: 'graph_update'
  graph: RecipeGraphData
}

export interface ErrorMessage {
  type: 'error'
  detail: string
}

export type ServerMessage = AgentEvent | AgentResponse | RecipeUpdate | GraphUpdate | ErrorMessage

export interface ChatMessage {
  id: string
  role: 'user' | 'agent'
  content: string
  timestamp: Date
}
