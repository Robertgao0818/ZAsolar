// hook_geid_curl.js — Frida agent for Allmapsoft GEID downloader.exe
//
// Captures every URL libcurl is asked to fetch, plus the headers/UA/referer
// associated with each handle. Emits two kinds of records via send():
//   { kind: "setopt_url", handle, url }            — fires the moment URL is set
//   { kind: "perform",    handle, state }          — fires right before request
//
// state = { URL, USERAGENT, REFERER, COOKIE, POSTFIELDS, PROXY, CUSTOMREQUEST,
//           HTTPHEADER: [string, ...] }
//
// libcurl option type tags (from curl.h):
//   CURLOPTTYPE_LONG         =     0
//   CURLOPTTYPE_OBJECTPOINT  = 10000   // string / data pointers
//   CURLOPTTYPE_FUNCTIONPOINT= 20000
//   CURLOPTTYPE_OFF_T        = 30000

const STRING_OPTS = {
  10002: "URL",
  10004: "PROXY",
  10015: "POSTFIELDS",
  10016: "REFERER",
  10018: "USERAGENT",
  10022: "COOKIE",
  10036: "CUSTOMREQUEST",
};
const CURLOPT_URL = 10002;
const CURLOPT_HTTPHEADER = 10023;

const handles = {}; // handle hex -> last-seen state

function readSlist(headPtr) {
  const out = [];
  let cur = headPtr;
  let safety = 0;
  while (cur && !cur.isNull() && safety < 256) {
    let dataPtr;
    try { dataPtr = cur.readPointer(); } catch (_) { break; }
    if (!dataPtr.isNull()) {
      try { out.push(dataPtr.readUtf8String()); } catch (_) {}
    }
    try { cur = cur.add(Process.pointerSize).readPointer(); } catch (_) { break; }
    safety++;
  }
  return out;
}

function emit(rec) {
  rec.t = Date.now();
  send(rec);
}

function findExport(modName, sym) {
  // Frida 17 removed Module.findExportByName; use the per-module API instead.
  const mod = Process.findModuleByName(modName);
  if (!mod) return null;
  try {
    return mod.findExportByName(sym);
  } catch (_) {
    return null;
  }
}

function installHooks() {
  const setoptAddr = findExport("libcurl.dll", "curl_easy_setopt");
  const performAddr = findExport("libcurl.dll", "curl_easy_perform");
  if (!setoptAddr || !performAddr) return false;

  console.log("[hook] curl_easy_setopt @ " + setoptAddr);
  console.log("[hook] curl_easy_perform @ " + performAddr);

  Interceptor.attach(setoptAddr, {
    onEnter(args) {
      const handle = args[0];
      const opt = args[1].toInt32();
      const val = args[2];
      const key = handle.toString();

      if (opt in STRING_OPTS) {
        let s = null;
        try { s = val.readUtf8String(); } catch (_) {}
        if (s !== null) {
          if (!handles[key]) handles[key] = {};
          handles[key][STRING_OPTS[opt]] = s;
          if (opt === CURLOPT_URL) {
            emit({ kind: "setopt_url", handle: key, url: s });
          }
        }
      } else if (opt === CURLOPT_HTTPHEADER) {
        try {
          const hs = readSlist(val);
          if (!handles[key]) handles[key] = {};
          handles[key].HTTPHEADER = hs;
        } catch (e) {
          console.error("[hook] slist walk failed: " + e);
        }
      }
    },
  });

  Interceptor.attach(performAddr, {
    onEnter(args) {
      const key = args[0].toString();
      const state = handles[key] || {};
      emit({ kind: "perform", handle: key, state });
    },
  });

  // Reset per-handle state when libcurl resets the handle
  const resetAddr = findExport("libcurl.dll", "curl_easy_reset");
  if (resetAddr) {
    Interceptor.attach(resetAddr, {
      onEnter(args) { delete handles[args[0].toString()]; },
    });
  }

  return true;
}

if (!installHooks()) {
  console.log("[hook] libcurl.dll not loaded yet, polling...");
  const tid = setInterval(() => {
    if (installHooks()) {
      clearInterval(tid);
      console.log("[hook] hooks installed");
    }
  }, 200);
} else {
  console.log("[hook] hooks installed");
}
