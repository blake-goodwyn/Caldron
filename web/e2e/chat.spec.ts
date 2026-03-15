import { test, expect } from '@playwright/test'
import { setupMockWebSocket } from './mock-ws'

test.describe('Chat Flow', () => {
  test.beforeEach(async ({ page }) => {
    // Install mock WebSocket before navigating
    await page.addInitScript(setupMockWebSocket, {})
    await page.goto('/')
    // Wait for mock WS to "connect"
    await page.waitForTimeout(200)
  })

  test('typing in input enables send button', async ({ page }) => {
    const input = page.getByTestId('chat-input')
    const sendBtn = page.getByTestId('chat-send')

    await expect(sendBtn).toBeDisabled()
    await input.fill('Make me a cake')
    await expect(sendBtn).toBeEnabled()
  })

  test('sending a message adds user bubble and clears input', async ({ page }) => {
    const input = page.getByTestId('chat-input')

    await input.fill('Make me a spicy miso ramen')
    await page.getByTestId('chat-send').click()

    // User message should appear
    await expect(page.getByTestId('message-user')).toBeVisible()
    await expect(page.getByTestId('message-user')).toContainText('spicy miso ramen')

    // Input should be cleared
    await expect(input).toHaveValue('')

    // Empty state should be gone
    await expect(page.getByTestId('chat-empty')).not.toBeVisible()
  })

  test('can send message with Enter key', async ({ page }) => {
    const input = page.getByTestId('chat-input')

    await input.fill('chocolate chip cookies')
    await input.press('Enter')

    await expect(page.getByTestId('message-user')).toBeVisible()
    await expect(input).toHaveValue('')
  })

  test('empty message is not sent', async ({ page }) => {
    const input = page.getByTestId('chat-input')

    // Whitespace-only should not enable the send button
    await input.fill('   ')
    const sendBtn = page.getByTestId('chat-send')
    await expect(sendBtn).toBeDisabled()
  })
})
