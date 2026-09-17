// @vitest-environment jsdom
import { act } from 'react'
import { createRoot } from 'react-dom/client'
import { expect, it, vi } from 'vitest'
import { OAuthProvidersCard } from './OAuthProvidersCard'

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true })
const api = vi.hoisted(() => ({
  getOAuthProviders: vi.fn(async () => ({ providers: [
    { id: 'nous', name: 'Nous Portal', flow: 'device_code', status: { logged_in: true } },
    { id: 'copilot', name: 'GitHub Copilot', flow: 'device_code', status: { logged_in: false } }
  ] })),
  startOAuthLogin: vi.fn(async () => ({ flow: 'device_code', session_id: 'copilot-test', user_code: 'TEST-CODE', verification_url: 'https://github.com/login/device', expires_in: 600 })),
  cancelOAuthSession: vi.fn(async () => ({}))
}))
vi.mock('@/lib/api', () => ({ api }))
vi.mock('@/i18n', () => ({ useI18n: () => ({ t: { common: { refresh: 'Refresh', close: 'Close' }, oauth: {
  sessionExpires: '{time}', providerLogins: 'Accounts', description: '{connected}/{total} connected', flowLabels: { device_code: 'Device code' }, connected: 'Connected', notConnected: 'Sign in', login: 'Sign in', disconnect: 'Disconnect', expiresIn: '{time}', connect: 'Connect', deviceInstructions: '', steps: {},
} } }) }))

it('uses direct Copilot device authentication alongside Nous without CLI login instructions', async () => {
  const node = document.createElement('div')
  document.body.append(node)
  const root = createRoot(node)
  const open = vi.spyOn(window, 'open').mockImplementation(() => null)
  try {
    await act(async () => root.render(<OAuthProvidersCard />))
    expect(node.textContent).toContain('Nous Portal')
    expect(node.textContent).toContain('GitHub Copilot')
    const login = [...node.querySelectorAll('button')].find(button => button.textContent === 'Sign in')!
    await act(async () => login.click())
    expect(api.startOAuthLogin).toHaveBeenCalledWith('copilot')
    expect(open).toHaveBeenCalledWith('https://github.com/login/device', '_blank', 'noopener,noreferrer')
    expect(node.textContent).not.toContain('copilot login')
  } finally {
    await act(async () => root.unmount())
    node.remove()
    open.mockRestore()
  }
})
