// @vitest-environment jsdom
import { act } from 'react'
import { createRoot } from 'react-dom/client'
import { afterEach, expect, it, vi } from 'vitest'
import ModelsPage from './ModelsPage'

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true })

const api = vi.hoisted(() => ({
  getModelsAnalytics: vi.fn(async () => ({ models: [], totals: { distinct_models: 0, total_sessions: 0 } })),
  getConfig: vi.fn(async () => ({})),
  getAuxiliaryModels: vi.fn(async () => ({ main: { provider: 'nous', model: 'hermes' }, tasks: [] })),
  getMoaModels: vi.fn(async () => null)
}))
vi.mock('@/lib/api', () => ({ api }))
vi.mock('@/contexts/usePageHeader', () => ({ usePageHeader: () => ({ setAfterTitle: () => {}, setEnd: () => {} }) }))
vi.mock('@/plugins', () => ({ PluginSlot: () => null }))
vi.mock('@/i18n', () => ({ useI18n: () => ({ t: { models: { modelsUsed: 'Models used' }, common: { refresh: 'Refresh' }, analytics: { totalSessions: 'Sessions' } } }) }))

const roots: ReturnType<typeof createRoot>[] = []
afterEach(async () => {
  for (const root of roots.splice(0)) await act(async () => root.unmount())
  document.body.innerHTML = ''
  vi.clearAllMocks()
})
it('does not request or expose removed Mixture of Agents settings', async () => {
  const node = document.createElement('div')
  document.body.append(node)
  const root = createRoot(node)
  roots.push(root)
  await act(async () => root.render(<ModelsPage />))
  expect(api.getMoaModels).not.toHaveBeenCalled()
  expect(node.textContent).not.toContain('Mixture of Agents')
})
