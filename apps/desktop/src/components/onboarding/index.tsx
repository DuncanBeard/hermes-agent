import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useRef, useState } from 'react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Input } from '@/components/ui/input'
import { Progress } from '@/components/ui/progress'
import { useI18n } from '@/i18n'
import { ChevronLeft, KeyRound, Loader2 } from '@/lib/icons'
import { isSubmitEnter } from '@/lib/ime'
import { isProviderSetupErrorMessage } from '@/lib/provider-setup-errors'
import { cn } from '@/lib/utils'
import { $desktopBoot, type DesktopBootState } from '@/store/boot'
import { $freeTierStatus, FREE_TIER_MODEL, freeTierSetupFailure } from '@/store/free-tier'
import { openFreeTierSignIn } from '@/store/free-tier-sign-in'
import { $introReveal, shouldPlayFirstRunIntro } from '@/store/intro-reveal'
import { $setupReadyTick } from '@/store/live-sync'
import { $localModelsEnabled } from '@/store/local-models-flag'
import {
  $desktopOnboarding,
  ackFreeTierIntro,
  clearFreeTierIntro,
  clearPendingProviderOAuth,
  closeManualOnboarding,
  confirmOnboardingModel,
  DEFAULT_MANUAL_ONBOARDING_REASON,
  DEFAULT_ONBOARDING_REASON,
  dismissFirstRunOnboarding,
  type OnboardingContext,
  peekPendingProviderOAuth,
  refreshOnboarding,
  saveOnboardingLocalEndpoint,
  setOnboardingMode,
  startManualOnboarding,
  startProviderOAuth
} from '@/store/onboarding'
import { $onboardingSurfaces, onboardingSurfaceActive } from '@/store/onboarding-presence'
import type { OAuthProvider } from '@/types/hermes'

import { FlowPanel, Status } from './flow'
import { FreeTierSetupNotice } from './free-tier-setup-notice'
import { DecodedLabel } from './glyph'
import { FeaturedProviderRow, LocalModelsProviderRow, ProviderRow, sortProviders } from './providers'

export {
  FeaturedProviderRow,
  KeyProviderRow,
  LocalModelsProviderRow,
  ProviderRow,
  providerTitle,
  sortProviders
} from './providers'

import { requestGatewayForProfile } from '@/store/gateway'

interface DesktopOnboardingOverlayProps {
  enabled: boolean
  onCompleted?: () => void
  profile: string
  requestGateway: OnboardingContext['requestGateway']
}

// Exit choreography, mirroring the gateway "connecting" overlay's timing:
// text-out (360ms: CONNECTED fades down, rest scrambles+fades) → hold (300ms)
// → surface-out (520ms, held back by [transition-delay:660ms]). Finalize after.
const ONBOARDING_EXIT_MS = 1180

export function DesktopOnboardingOverlay({
  enabled,
  onCompleted,
  profile,
  requestGateway
}: DesktopOnboardingOverlayProps) {
  const { t } = useI18n()
  const onboarding = useStore($desktopOnboarding)
  const boot = useStore($desktopBoot)
  const introReveal = useStore($introReveal)
  useStore($onboardingSurfaces)
  const onCompletedRef = useRef(onCompleted)
  onCompletedRef.current = onCompleted
  const targetProfile = onboarding.targetProfile ?? profile

  // Async flows retain the initiating route even after the overlay closes.
  const ctx = useMemo<OnboardingContext>(
    () => ({
      profile: targetProfile,
      requestGateway: onboarding.targetProfile
        ? (method, params) => requestGatewayForProfile(targetProfile, method, params)
        : requestGateway,
      onCompleted: () => onCompletedRef.current?.()
    }),
    [onboarding.targetProfile, targetProfile, requestGateway]
  )

  // Cinematic exit on "Begin": dissolve the panel + overlay (revealing the chat
  // behind), THEN finalize so the unmount lands after the fade — mirrors the
  // connecting overlay's exit choreography instead of cutting instantly.
  const [leaving, setLeaving] = useState(false)

  const finalizeOnboarding = () => {
    if (leaving) {
      return
    }

    const reduce = typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches

    if (reduce) {
      confirmOnboardingModel(ctx)

      return
    }

    setLeaving(true)
    window.setTimeout(() => confirmOnboardingModel(ctx), ONBOARDING_EXIT_MS)
  }

  // The free-tier intro's three doors share one exit: consume the notice, play
  // the same dissolve, then run whatever the door opens onto. `after` runs at
  // the END so a sign-in dialog or provider picker never appears behind a
  // still-fading overlay.
  const dismissFreeTierIntro = async (after?: () => void) => {
    if (leaving) {
      return
    }

    // The screen is keyed on the backend's notice flag: only a recorded ack takes it down.
    // A failed ack leaves it in place for another try rather than hiding the only notice.
    if (!(await ackFreeTierIntro(ctx))) {
      return
    }

    const reduce = typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches

    if (reduce) {
      clearFreeTierIntro()
      after?.()

      return
    }

    setLeaving(true)
    window.setTimeout(() => {
      setLeaving(false)
      clearFreeTierIntro()
      after?.()
    }, ONBOARDING_EXIT_MS)
  }

  useEffect(() => {
    if (enabled || onboarding.requested) {
      void refreshOnboarding(ctx)
    }
  }, [ctx, enabled, onboarding.requested])

  // The boot bootstrap re-announces `setup.ready` when a background retry of
  // the free-tier set-up succeeds after a failed first attempt. A picker that
  // is up only because that set-up failed re-checks readiness and gives way
  // on its own. An untouched picker only: a manual open, a provider flow in
  // progress, or the API-key form (which leaves the flow idle while the user
  // types) is left alone, and the check is repeated after the readiness
  // round so a key form opened in the meantime survives too.
  useEffect(
    () =>
      $setupReadyTick.listen(() => {
        const untouched = () => {
          const current = $desktopOnboarding.get()

          return (
            !current.manual &&
            current.configured === false &&
            current.flow.status === 'idle' &&
            current.mode === 'oauth' &&
            !current.localEndpoint
          )
        }

        if (untouched()) {
          void refreshOnboarding(ctx, untouched)
        }
      }),
    [ctx]
  )
  const freeTierStatus = useStore($freeTierStatus)
  const setupFailure = !onboarding.manual ? freeTierSetupFailure(freeTierStatus) : null

  // When the Providers settings page asked to connect a specific provider, the
  // store stashed its id. Once the provider list has loaded and we're back at
  // an idle picker, launch that exact OAuth flow so the user lands directly in
  // sign-in instead of the picker they just came from.
  useEffect(() => {
    if (!onboarding.manual || onboarding.providers === null || onboarding.flow.status !== 'idle') {
      return
    }

    const pendingId = peekPendingProviderOAuth()

    if (!pendingId) {
      return
    }

    const provider = onboarding.providers.find(p => p.id === pendingId)

    if (provider) {
      // Only clear once we've committed to launching it, so a failed/empty
      // provider fetch doesn't silently drop the hand-off.
      clearPendingProviderOAuth()
      void startProviderOAuth(provider, ctx)
    } else if (onboarding.providers.length > 0) {
      // The list loaded but the id isn't a real provider — drop the stale
      // hand-off. An empty list means the fetch isn't ready yet, so keep it
      // and let a later refresh retry.
      clearPendingProviderOAuth()
    }
  }, [ctx, onboarding.flow.status, onboarding.manual, onboarding.providers])

  if (
    !onboarding.manual &&
    (introReveal.phase !== 'hidden' || onboardingSurfaceActive() || shouldPlayFirstRunIntro(onboarding.firstRunSkipped))
  ) {
    return null
  }

  // Mount from frame 1 so we replace the boot overlay seamlessly. The
  // configured field stays null until the runtime check resolves; only then
  // do we know whether to dismiss (true) or surface the picker (false).
  // EXCEPTION: manual mode (user opened the selector from a working app to
  // add/switch a provider) shows the overlay regardless of configured state.
  if (onboarding.configured === true && !onboarding.manual && !onboarding.freeTierReady) {
    return null
  }

  // The user chose "I'll choose a provider later" on first run. Stay out of the
  // way on every subsequent launch — they re-enter via Settings → Providers
  // (manual mode), which sets manual=true and bypasses this gate.
  if (onboarding.firstRunSkipped && !onboarding.manual && !onboarding.freeTierReady) {
    return null
  }

  const { flow } = onboarding
  // Show the launch reason only when it's a meaningful, caller-supplied prompt —
  // suppress the generic defaults (useless noise) and provider-setup errors
  // (those are surfaced by FlowPanel, not as a banner).
  const rawReason = onboarding.reason?.trim() || null

  // When the free tier itself failed to set up, its own notice explains the
  // picker; the runtime check's technical reason ("No usable credentials
  // found for nous.") would only restate it in the wrong words.
  const reason =
    rawReason &&
    !setupFailure &&
    !isProviderSetupErrorMessage(rawReason) &&
    rawReason !== DEFAULT_ONBOARDING_REASON &&
    rawReason !== DEFAULT_MANUAL_ONBOARDING_REASON
      ? rawReason
      : null

  // In manual mode the app is already configured, so the flow is "ready"
  // immediately — no runtime gate needed. Otherwise wait for the readiness
  // check (configured === false) before showing the picker.
  // The free-tier intro owns the overlay while it is up: the app is already
  // configured, so there is no picker to show and no runtime gate to wait on.
  // A manual open (the user asked for the picker) outranks it.
  const freeTierIntro = onboarding.freeTierReady && !onboarding.manual && flow.status === 'idle'
  const ready = freeTierIntro || onboarding.manual || (enabled && onboarding.configured === false)
  const showPicker = !freeTierIntro && (flow.status === 'idle' || flow.status === 'success')
  // The final "you're in" screen drops the card chrome and floats centered on
  // the surface — same bare, cinematic treatment as the connecting overlay.
  const bare = ready && (freeTierIntro || (!showPicker && flow.status === 'confirming_model'))

  return (
    <div
      className={cn(
        'fixed inset-0 z-(--z-onboarding) flex items-center justify-center bg-(--ui-chat-surface-background) p-6 transition-opacity duration-[520ms] ease-out',
        // On the bare confirm screen, hold the surface (text-out + hold) so the
        // per-element exit plays before it dissolves.
        bare && leaving ? '[transition-delay:660ms]' : '',
        leaving ? 'pointer-events-none opacity-0' : 'opacity-100'
      )}
      // Masks the whole app until onboarding finishes — must stay filled under
      // window glass or the shell shows through. Contract:
      // `[data-glass-opaque]` in styles.css.
      data-glass-opaque=""
    >
      <div
        className={cn(
          'relative w-full max-w-[45rem] transition-all duration-500 ease-out',
          bare
            ? ''
            : 'overflow-hidden rounded-xl border border-(--stroke-nous) bg-(--ui-chat-bubble-background) shadow-nous',
          // Bare confirm screen orchestrates its own per-element exit; the
          // carded states use the simple lift/blur dissolve.
          leaving && !bare
            ? '-translate-y-1 scale-[0.985] opacity-0 blur-[2px]'
            : 'translate-y-0 scale-100 opacity-100 blur-0'
        )}
      >
        {showPicker || !ready ? <Header /> : null}
        {onboarding.manual ? (
          <Button
            aria-label={t.common.close}
            className="absolute right-3 top-3 z-10 text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground"
            onClick={() => closeManualOnboarding()}
            size="icon-sm"
            variant="ghost"
          >
            <Codicon name="close" size="1rem" />
          </Button>
        ) : null}
        <div className="grid gap-3 p-5">
          {reason ? <ReasonNotice reason={reason} /> : null}
          {ready && showPicker && !freeTierIntro && !onboarding.manual ? <FreeTierSetupNotice ctx={ctx} /> : null}
          {ready ? (
            freeTierIntro ? (
              <FreeTierReadyPanel leaving={leaving} onDismiss={dismissFreeTierIntro} />
            ) : showPicker ? (
              <Picker ctx={ctx} />
            ) : (
              <FlowPanel ctx={ctx} flow={flow} leaving={leaving} onBegin={finalizeOnboarding} />
            )
          ) : (
            <Preparing boot={boot} />
          )}
        </div>
      </div>
    </div>
  )
}

/**
 * The one-time free-tier welcome, shown when the free tier is what serves this
 * user. Bare and centered like the model-confirm screen it stands in for: this
 * IS their "you're in" moment, so it names the route, its model and its price,
 * and offers the two ways out of it (a real account, or a provider of their
 * own) without making either the default.
 */
function FreeTierReadyPanel({
  leaving,
  onDismiss
}: {
  leaving: boolean
  onDismiss: (after?: () => void) => Promise<void>
}) {
  const { t } = useI18n()
  const copy = t.freeTier

  return (
    <div className="grid place-items-center gap-7 py-6 text-center">
      <DecodedLabel leaving={leaving} text={copy.readyTitle} />

      <div
        className={cn(
          'grid justify-items-center gap-1.5 transition duration-[360ms] ease-out',
          leaving ? 'opacity-0 saturate-0' : 'opacity-100 saturate-100'
        )}
      >
        <div className="flex items-center gap-2">
          <span className="font-mono text-[0.625rem] uppercase tracking-[0.2em] text-muted-foreground">
            {t.onboarding.defaultModel}
          </span>
          <Badge size="xs" variant="success">
            {t.onboarding.freeTier}
          </Badge>
        </div>
        <p className="font-mono text-base">{FREE_TIER_MODEL}</p>
        <p className="font-mono text-xs text-muted-foreground">{copy.readyCaption}</p>
      </div>

      <div
        className={cn(
          'grid justify-items-center gap-2 transition duration-[360ms] ease-out',
          leaving ? 'opacity-0 saturate-0' : 'opacity-100 saturate-100'
        )}
      >
        <Button onClick={() => void onDismiss()} type="button">
          {copy.begin}
        </Button>
        <Button onClick={() => void onDismiss(() => openFreeTierSignIn())} size="xs" type="button" variant="text">
          {copy.signInInstead}
        </Button>
        <Button
          onClick={() => void onDismiss(() => startManualOnboarding(null))}
          size="xs"
          type="button"
          variant="text"
        >
          {copy.otherProviders}
        </Button>
      </div>
    </div>
  )
}

// The launch reason is a prompt ("why am I seeing this"), not an error. Only
// rendered for meaningful caller-supplied reasons (defaults are filtered out
// upstream), so it never shows the generic "no provider configured" noise.
function ReasonNotice({ reason }: { reason: string }) {
  return (
    <div className="rounded-2xl border border-(--ui-stroke-tertiary) bg-(--ui-bg-tertiary)/40 px-4 py-3 text-sm text-muted-foreground">
      {reason}
    </div>
  )
}

function Preparing({ boot }: { boot: DesktopBootState }) {
  const { t } = useI18n()
  const progress = Math.max(2, Math.min(100, Math.round(boot.progress)))
  const hasError = Boolean(boot.error)
  const installing = boot.phase.startsWith('runtime.')

  return (
    <div className="grid gap-3" role="status">
      <p className="text-sm text-muted-foreground">
        {installing ? t.onboarding.preparingInstall : t.onboarding.starting}
      </p>
      <Progress
        aria-label={installing ? t.onboarding.preparingInstall : t.onboarding.starting}
        destructive={hasError}
        size="lg"
        value={progress / 100}
      />
      <div className="flex items-center justify-between gap-3 text-xs text-muted-foreground">
        <span className="truncate">{boot.message}</span>
        <span>{progress}%</span>
      </div>
      {hasError ? <p className="text-xs text-destructive">{boot.error}</p> : null}
    </div>
  )
}

function Header() {
  const { t } = useI18n()

  return (
    <div className="bg-(--ui-chat-bubble-background) px-5 pt-5 pb-1">
      <h2 className="text-[0.9375rem] font-semibold tracking-tight">{t.onboarding.headerTitle}</h2>
      <p className="mt-1 max-w-xl text-[0.8125rem] leading-5 text-(--ui-text-tertiary)">{t.onboarding.headerDesc}</p>
    </div>
  )
}

export const FEATURED_ID = 'nous'

export function Picker({ ctx }: { ctx: OnboardingContext }) {
  const { t } = useI18n()
  const { localEndpoint, manual, mode, providers } = useStore($desktopOnboarding)
  const ordered = useMemo(() => sortProviders(providers ?? []), [providers])

  if (localEndpoint || mode === 'apikey') {
    return (
      <CustomEndpointForm
        canGoBack={!localEndpoint}
        onBack={() => setOnboardingMode('oauth')}
        onSave={(url, apiKey) => saveOnboardingLocalEndpoint(url, apiKey, ctx)}
      />
    )
  }

  if (providers === null) {
    return <Status>{t.onboarding.lookingUpProviders}</Status>
  }

  const select = (provider: OAuthProvider) => void startProviderOAuth(provider, ctx)

  const openLocalModels = () => {
    if (manual) {
      closeManualOnboarding()
    } else {
      dismissFirstRunOnboarding()
    }

    window.location.hash = '#/settings?tab=providers&pview=local'
  }

  return (
    <div className="grid gap-2">
      {ordered.map(provider =>
        provider.id === FEATURED_ID ? (
          <FeaturedProviderRow key={provider.id} onSelect={select} provider={provider} />
        ) : (
          <ProviderRow key={provider.id} onSelect={select} provider={provider} />
        )
      )}
      {$localModelsEnabled.get() ? <LocalModelsProviderRow onClick={openLocalModels} /> : null}
      <Button onClick={() => setOnboardingMode('apikey')} size="sm" variant="outline">
        {t.settings.providers.localEndpoint.title}
      </Button>
      {manual ? null : <ChooseLaterLink />}
    </div>
  )
}

// "I'll choose a provider later" — dismisses the first-run picker and persists
// the skip so it never re-nags. The user connects a provider any time from
// Settings → Providers. Rendered only on the unconfigured first-run flow.
function ChooseLaterLink() {
  const { t } = useI18n()

  return (
    <Button className="font-medium" onClick={() => dismissFirstRunOnboarding()} size="xs" type="button" variant="text">
      {t.onboarding.chooseLater}
    </Button>
  )
}

function CustomEndpointForm({
  canGoBack,
  onBack,
  onSave
}: {
  canGoBack: boolean
  onBack: () => void
  onSave: (url: string, apiKey: string) => Promise<{ message?: string; ok: boolean }>
}) {
  const { t } = useI18n()
  const [url, setUrl] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const submit = async () => {
    if (!url.trim() || saving) {
      return
    }

    setSaving(true)
    setError(null)

    try {
      const result = await onSave(url, apiKey)

      if (result.ok) {
        setUrl('')
        setApiKey('')
      } else {
        setError(result.message ?? t.onboarding.couldNotSave)
      }
    } catch (error) {
      setError(error instanceof Error ? error.message : t.onboarding.couldNotSave)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="grid gap-3">
      {canGoBack && (
        <Button onClick={onBack} size="xs" variant="text">
          <ChevronLeft />
          {t.onboarding.backToSignIn}
        </Button>
      )}
      <p className="text-sm text-muted-foreground">{t.onboarding.apiKeyOptions.local.description}</p>
      <Input
        aria-label="Endpoint URL"
        autoComplete="off"
        autoFocus
        onChange={e => setUrl(e.target.value)}
        onKeyDown={e => isSubmitEnter(e) && void submit()}
        placeholder="http://127.0.0.1:8000/v1"
        value={url}
      />
      <Input
        autoComplete="off"
        onChange={e => setApiKey(e.target.value)}
        onKeyDown={e => isSubmitEnter(e) && void submit()}
        placeholder={t.onboarding.localApiKeyPlaceholder}
        type="password"
        value={apiKey}
      />
      {error && (
        <p className="text-xs text-destructive" role="alert">
          {error}
        </p>
      )}
      <Button disabled={!url.trim() || saving} onClick={() => void submit()}>
        {saving ? <Loader2 className="animate-spin" /> : <KeyRound />}
        {saving ? t.onboarding.connecting : t.common.connect}
      </Button>
    </div>
  )
}
