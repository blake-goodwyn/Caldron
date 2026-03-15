import { test, expect } from '@playwright/test'

test.describe('Caldron App', () => {
  test('loads the app with header and layout', async ({ page }) => {
    await page.goto('/')
    // Header h1 contains emoji + "Caldron"
    await expect(page.locator('header h1')).toContainText('Caldron')
    await expect(page.getByText('AI Recipe Development')).toBeVisible()
  })

  test('shows empty states on load', async ({ page }) => {
    await page.goto('/')
    // Chat empty state
    await expect(page.getByTestId('chat-empty')).toBeVisible()
    // Recipe empty state (default tab)
    await expect(page.getByTestId('recipe-empty')).toBeVisible()
  })

  test('chat input is present', async ({ page }) => {
    await page.goto('/')
    const input = page.getByTestId('chat-input')
    await expect(input).toBeVisible()
    await expect(input).toHaveAttribute('placeholder', 'Describe a recipe idea...')
  })

  test('send button is disabled when disconnected', async ({ page }) => {
    await page.goto('/')
    // Without a backend, WebSocket won't connect, so send is disabled
    const sendBtn = page.getByTestId('chat-send')
    await expect(sendBtn).toBeDisabled()
  })

  test('can switch between recipe and graph tabs', async ({ page }) => {
    await page.goto('/')
    // Recipe tab is active by default
    await expect(page.getByTestId('recipe-empty')).toBeVisible()

    // Switch to graph tab
    await page.getByTestId('tab-graph').click()
    await expect(page.getByTestId('graph-empty')).toBeVisible()

    // Switch back to recipe tab
    await page.getByTestId('tab-recipe').click()
    await expect(page.getByTestId('recipe-empty')).toBeVisible()
  })

  test('shows connection status indicator', async ({ page }) => {
    await page.goto('/')
    await expect(page.getByTestId('connection-status')).toBeVisible()
  })
})
