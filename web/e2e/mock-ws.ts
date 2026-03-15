/**
 * Shared MockWebSocket setup for Playwright e2e tests.
 * Call page.addInitScript(setupMockWebSocket, options) before page.goto().
 */

export interface MockWSOptions {
  /** Messages the mock server sends in response to a user_message */
  responses?: object[]
}

/**
 * Replaces window.WebSocket with a mock that auto-connects
 * and optionally sends canned responses when a user_message is received.
 */
export function setupMockWebSocket(options: MockWSOptions = {}) {
  class MockWebSocket extends EventTarget {
    static CONNECTING = 0
    static OPEN = 1
    static CLOSING = 2
    static CLOSED = 3
    readyState = MockWebSocket.OPEN
    url: string
    CONNECTING = 0
    OPEN = 1
    CLOSING = 2
    CLOSED = 3
    binaryType: BinaryType = 'blob'
    bufferedAmount = 0
    extensions = ''
    protocol = ''
    onopen: ((ev: Event) => void) | null = null
    onclose: ((ev: CloseEvent) => void) | null = null
    onmessage: ((ev: MessageEvent) => void) | null = null
    onerror: ((ev: Event) => void) | null = null

    constructor(url: string | URL, _protocols?: string | string[]) {
      super()
      this.url = url.toString()
      setTimeout(() => {
        this.readyState = MockWebSocket.OPEN
        const openEvent = new Event('open')
        this.onopen?.(openEvent)
        this.dispatchEvent(openEvent)
      }, 50)
    }

    send(data: string) {
      const msg = JSON.parse(data)
      if (msg.type === 'user_message' && options.responses) {
        options.responses.forEach((resp, i) => {
          setTimeout(() => {
            const event = new MessageEvent('message', { data: JSON.stringify(resp) })
            this.onmessage?.(event)
            this.dispatchEvent(event)
          }, (i + 1) * 100)
        })
      }
    }

    close() {
      this.readyState = MockWebSocket.CLOSED
    }
  }

  ;(window as any).WebSocket = MockWebSocket
}
