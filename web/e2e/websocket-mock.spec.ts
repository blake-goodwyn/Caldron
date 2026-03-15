import { test, expect } from '@playwright/test'
import { setupMockWebSocket } from './mock-ws'

const MOCK_RECIPE = {
  name: 'Spicy Miso Ramen',
  ingredients: [
    { name: 'ramen noodles', quantity: 200, unit: 'g' },
    { name: 'white miso paste', quantity: 3, unit: 'tbsp' },
    { name: 'chili oil', quantity: 2, unit: 'tsp' },
    { name: 'soft-boiled egg', quantity: 2, unit: null },
  ],
  instructions: [
    'Bring 4 cups of water to a boil.',
    'Dissolve miso paste in hot water.',
    'Cook ramen noodles according to package directions.',
    'Assemble bowls and top with chili oil and egg.',
  ],
  tags: ['japanese', 'spicy', 'soup'],
  sources: ['https://example.com/ramen'],
}

const MOCK_GRAPH = {
  foundational_recipe_node: 'node-2',
  nodes: [
    { node_id: 'node-1', recipe: { ...MOCK_RECIPE, name: 'Miso Ramen v1' } },
    { node_id: 'node-2', recipe: MOCK_RECIPE },
  ],
  edges: [{ source: 'node-1', target: 'node-2' }],
}

const MOCK_RESPONSES = [
  { type: 'agent_event', agent: 'Tavily', status: 'working', content: null },
  { type: 'agent_event', agent: 'Tavily', status: 'done', content: 'Found recipes' },
  { type: 'agent_response', content: 'Here is your Spicy Miso Ramen recipe!' },
  { type: 'recipe_update', recipe: MOCK_RECIPE },
  { type: 'graph_update', graph: MOCK_GRAPH },
]

test.describe('Full Chat Flow with Mock WebSocket', () => {
  test('sends message, receives response, renders recipe card and graph', async ({ page }) => {
    await page.addInitScript(setupMockWebSocket, { responses: MOCK_RESPONSES })
    await page.goto('/')
    await page.waitForTimeout(200)

    // Send a message
    await page.getByTestId('chat-input').fill('Make me a spicy miso ramen')
    await page.getByTestId('chat-send').click()

    // User message appears
    await expect(page.getByTestId('message-user')).toBeVisible()

    // Agent response arrives
    await expect(page.getByTestId('message-agent')).toBeVisible({ timeout: 3000 })
    await expect(page.getByTestId('message-agent')).toContainText('Spicy Miso Ramen')

    // Recipe card populates
    await expect(page.getByTestId('recipe-card')).toBeVisible({ timeout: 3000 })
    await expect(page.getByTestId('recipe-name')).toHaveText('Spicy Miso Ramen')

    // Verify ingredients rendered (scoped to recipe card)
    const card = page.getByTestId('recipe-card')
    await expect(card.getByText('ramen noodles').first()).toBeVisible()
    await expect(card.getByText('white miso paste')).toBeVisible()

    // Verify instructions rendered
    await expect(card.getByText('Bring 4 cups of water to a boil.')).toBeVisible()

    // Verify tags rendered
    await expect(card.getByText('japanese')).toBeVisible()
    await expect(card.getByText('spicy').first()).toBeVisible()

    // Switch to graph tab
    await page.getByTestId('tab-graph').click()
    await expect(page.getByTestId('recipe-graph')).toBeVisible({ timeout: 3000 })
  })

  test('agent indicator shows during processing', async ({ page }) => {
    // Only send the working event, delay the response
    await page.addInitScript(setupMockWebSocket, {
      responses: [
        { type: 'agent_event', agent: 'Tavily', status: 'working', content: null },
      ],
    })
    await page.goto('/')
    await page.waitForTimeout(200)

    await page.getByTestId('chat-input').fill('test')
    await page.getByTestId('chat-send').click()

    // Agent indicator should appear
    await expect(page.getByTestId('agent-indicator')).toBeVisible({ timeout: 2000 })
    await expect(page.getByTestId('agent-indicator')).toContainText('Searching web')
  })
})
