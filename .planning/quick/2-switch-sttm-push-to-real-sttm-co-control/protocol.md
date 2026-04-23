# Real STTM Bani Controller protocol

Reverse-engineered from `https://www.sikhitothemax.org/assets/js/app-42ac2e74385481808980.js`
(the JS bundle loaded by `sttm.co/control`).

## Transport

**Socket.IO** client. Dependency on loader: the page injects
`//api.sikhitothemax.org/socket.io/socket.io.js` (= server-advertised client lib).

Connection:

```js
window.io("//api.sikhitothemax.org/" + namespaceString)
```

- `namespaceString` is the **Sangat Sync code** (regex `[A-Z,a-z]{3}-[A-Z,a-z]{3}`).
  Visible in STTM Desktop → Settings → Sync. Example seen today: `SPW-OEG`.
- Rotates per STTM Desktop launch — not persistent.
- URL form for Python: `https://api.sikhitothemax.org` with `socketio_path="/socket.io"` and `namespaces=["/SPW-OEG"]`.

## Pre-connect handshake (HTTP)

The web client hits `GET //api.sikhitothemax.org/sync/join/<CODE>` first to
validate the code and fetch existing session data. It's not strictly required
for control — `request-control` over socket.io is the real auth — but it's a
cheap way to get a clean "bad code" error before opening a socket.

## Socket.IO events

### Outbound (client → relay → STTM Desktop)

All emits use event name `"data"`. Payload variants (exact shapes, from bundle):

```json
// Auth / request control — FIRST emit after connecting
{"host":"sttm-web", "type":"request-control", "pin": 3879}

// Advance to a specific verse within an open shabad (this is the one we need)
{"host":"sttm-web", "type":"shabad",   "pin": 3879, "shabadId": 628, "verseId": 30545, "gurmukhi": "<text>"}

// Bani (nitnem-style) — unused by us
{"host":"sttm-web", "type":"bani",     "pin": 3879, "baniId": 1}

// Ceremony — unused by us
{"host":"sttm-web", "type":"ceremony", "pin": 3879, "ceremonyId": 3, "verseId": 26106}

// Plain-text slide — unused by us
{"host":"sttm-web", "type":"text",     "pin": 3879, "text": "vwihgurU", "isGurmukhi": true, "isAnnouncement": true}

// Generic pangti click (supports ceremony/bani context and lineCount index)
{"host":"sttm-web", "type": "<resultType>", "pin": 3879, "ceremonyId": null, "baniId": null,
 "shabadId": 628, "verseId": 30545, "lineCount": 5}

// Live settings push
{"host":"sttm-web", "type":"settings", "pin": 3879, "settings": { ... }}
```

**Key finding:** the "advance line" payload is `type:"shabad"` with `shabadId + verseId + gurmukhi`. No `homeId`. No `lineCount`. The separate `handlePanktiClick` variant uses `lineCount` (1-based position) but only for ceremony/bani contexts.

### Inbound (relay → client)

```js
socket.on("data", (e) => {
  if (e.type === "response-control") {
    if (e.success) { /* authenticated */ }
    else { /* pin mismatch */ }
  }
})
socket.on("close", ...)
```

**PIN auth flow:** after emitting `request-control`, wait for a `"data"` event
with `type === "response-control"` and `success === true`. If `success === false`,
the PIN is wrong. Timeout ~3 s is safe.

## Our implementation notes

1. Dep: `python-socketio[client]` (pulls `websocket-client` + `engineio`).
2. State: the socket is **persistent** — open once on app launch, keep open,
   emit `request-control` once, then `type:"shabad"` payloads per push.
   Reconnect on `close` with the same namespace + pin.
3. Code persistence: save `sttm_sync_code` to `~/.surt/config.json` next to
   `sttm_pin`. User updates it when STTM rotates the code.
4. UI: add a "Sync code" text input beside the PIN field. Placeholder
   `ABC-XYZ`, regex `^[A-Za-z]{3}-[A-Za-z]{3}$`. Uppercase on blur.
5. `discover()` can be replaced by a one-shot `GET /sync/join/<CODE>` + a
   boolean for whether the socket is connected + authenticated. No more
   localhost port probing.
6. PIN is an **integer** in the emit payload (`parseInt(t)` in the bundle),
   not a string. Cast before emitting.
