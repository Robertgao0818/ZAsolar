// hook_geid_ssl.js — Frida agent for capturing HTTP requests from Allmapsoft
// GEID downloader.exe via SSL_write (Indy/OpenSSL path) and ws2_32!send fallback.
//
// GEID 6.48 ships libcurl.dll but tile fetching actually goes through Delphi
// Indy components linked against OpenSSL 1.0 (ssleay32/libeay32). HTTPS request
// payloads can be captured at SSL_write() right before they're encrypted.

const HTTP_VERBS = ["GET ", "POST ", "PUT ", "HEAD ", "DELETE ", "OPTIONS "];
const handles = {}; // ssl_ptr.toString() -> { expectBody: int, lastReq: string }

function looksLikeHttp(s) {
  if (!s) return false;
  for (const v of HTTP_VERBS) if (s.startsWith(v)) return true;
  return false;
}

function readAscii(ptr, len) {
  // Strict: returns null if buffer contains any non-printable byte.
  // Used by SSL_write where HTTP requests are pure ASCII end-to-end.
  if (!ptr || ptr.isNull()) return null;
  const cap = Math.min(len, 4096);
  try {
    const bytes = ptr.readByteArray(cap);
    if (!bytes) return null;
    let s = "";
    const view = new Uint8Array(bytes);
    for (let i = 0; i < view.length; i++) {
      const b = view[i];
      if (b === 0) break;
      if (b < 9 || (b > 13 && b < 32) || b > 126) return null;
      s += String.fromCharCode(b);
    }
    return s;
  } catch (_) { return null; }
}

function readAsciiPrefix(ptr, len) {
  // Lenient: returns the printable prefix up to the first non-printable byte.
  // Lets us detect HTTP/ headers even when followed by binary body in the same buffer.
  if (!ptr || ptr.isNull()) return "";
  const cap = Math.min(len, 1024);
  try {
    const bytes = ptr.readByteArray(cap);
    if (!bytes) return "";
    let s = "";
    const view = new Uint8Array(bytes);
    for (let i = 0; i < view.length; i++) {
      const b = view[i];
      if (b === 0) break;
      if (b < 9 || (b > 13 && b < 32) || b > 126) break;
      s += String.fromCharCode(b);
    }
    return s;
  } catch (_) { return ""; }
}

function readHex(ptr, len) {
  if (!ptr || ptr.isNull()) return null;
  const cap = Math.min(len, 8192);
  try {
    const bytes = ptr.readByteArray(cap);
    if (!bytes) return null;
    const view = new Uint8Array(bytes);
    let s = "";
    for (let i = 0; i < view.length; i++) {
      s += view[i].toString(16).padStart(2, "0");
    }
    return s;
  } catch (_) { return null; }
}

function parseContentLength(headers) {
  const m = headers.match(/[Cc]ontent-[Ll]ength:\s*(\d+)/);
  return m ? parseInt(m[1], 10) : 0;
}

function emit(rec) {
  rec.t = Date.now();
  send(rec);
}

function findExport(modName, sym) {
  const mod = Process.findModuleByName(modName);
  if (!mod) return null;
  try { return mod.findExportByName(sym); } catch (_) { return null; }
}

function hookSslWrite(modName) {
  const addr = findExport(modName, "SSL_write");
  if (!addr) return false;
  console.log("[hook] " + modName + "!SSL_write @ " + addr);
  Interceptor.attach(addr, {
    onEnter(args) {
      // SSL_write(SSL *ssl, const void *buf, int num)
      const sslPtr = args[0].toString();
      const buf = args[1];
      const num = args[2].toInt32();
      if (num <= 0 || num > 65536) return;

      const ascii = readAscii(buf, num);
      if (looksLikeHttp(ascii)) {
        // Request headers — parse Content-Length so the next write can be flagged as body
        const cl = parseContentLength(ascii);
        handles[sslPtr] = { expectBody: cl, lastReq: ascii.split("\r\n")[0] };
        emit({ kind: "ssl_write", mod: modName, len: num, data: ascii });
        return;
      }

      // Maybe a request body following a recent POST/PUT?
      const state = handles[sslPtr];
      if (state && state.expectBody > 0) {
        const hex = readHex(buf, num);
        emit({
          kind: "ssl_write_body",
          mod: modName,
          len: num,
          hex: hex,
          for_req: state.lastReq,
        });
        state.expectBody = Math.max(0, state.expectBody - num);
      }
    },
  });
  return true;
}

function hookSslRead(modName) {
  const addr = findExport(modName, "SSL_read");
  if (!addr) return false;
  console.log("[hook] " + modName + "!SSL_read @ " + addr);
  Interceptor.attach(addr, {
    onEnter(args) {
      this.sslPtr = args[0].toString();
      this.buf = args[1];
    },
    onLeave(retval) {
      const n = retval.toInt32();
      if (n <= 0 || n > 65536) return;
      const state = handles[this.sslPtr] || {};
      // Always dump full hex (truncated to 8KB by readHex) so we never lose body bytes.
      // ascii_prefix is a human-readable preview of the printable prefix and is also
      // used by post-processing to detect HTTP/ response headers within mixed-content
      // chunks (e.g. headers + binary body in the same SSL_read return).
      const hex = readHex(this.buf, n);
      const prefix = readAsciiPrefix(this.buf, n);
      emit({
        kind: "ssl_read",
        mod: modName,
        len: n,
        hex: hex,
        ascii_prefix: prefix,
        for_req: state.lastReq || null,
      });
    },
  });
  return true;
}

function hookSocketSend() {
  // ws2_32!send(SOCKET s, const char* buf, int len, int flags)
  const addr = findExport("WS2_32.dll", "send") || findExport("ws2_32.dll", "send");
  if (!addr) return false;
  console.log("[hook] ws2_32!send @ " + addr);
  Interceptor.attach(addr, {
    onEnter(args) {
      const buf = args[1];
      const len = args[2].toInt32();
      if (len <= 0 || len > 65536) return;
      const s = readAscii(buf, len);
      if (looksLikeHttp(s)) {
        emit({ kind: "tcp_send", len: len, data: s });
      }
    },
  });
  return true;
}

function installHooks() {
  let count = 0;
  if (hookSslWrite("ssleay32.dll"))   count++;
  if (hookSslWrite("libssl-1_1.dll")) count++;
  if (hookSslRead("ssleay32.dll"))    count++;
  if (hookSslRead("libssl-1_1.dll"))  count++;
  if (hookSocketSend()) count++;
  return count > 0;
}

if (!installHooks()) {
  console.log("[hook] no SSL/socket libs loaded yet, polling...");
  const tid = setInterval(() => {
    if (installHooks()) {
      clearInterval(tid);
      console.log("[hook] hooks installed");
    }
  }, 200);
} else {
  console.log("[hook] hooks installed");
}
