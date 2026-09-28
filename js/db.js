// IndexedDB 封装：持久化动画剪辑与 UI 设置。
// 主线程与 Web Worker 均可使用（IndexedDB 在 Worker 中可用）。
const DB_NAME = 'skeletal-anim-demo';
const DB_VERSION = 1;

export function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains('clips')) db.createObjectStore('clips', { keyPath: 'name' });
      if (!db.objectStoreNames.contains('settings')) db.createObjectStore('settings', { keyPath: 'key' });
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function tx(db, store, mode, fn) {
  return new Promise((resolve, reject) => {
    const t = db.transaction(store, mode);
    const s = t.objectStore(store);
    const result = fn(s);
    t.oncomplete = () => resolve(result && result._value !== undefined ? result._value : result);
    t.onerror = () => reject(t.error);
  });
}

export function dbGet(db, store, key) {
  return new Promise((resolve, reject) => {
    const req = db.transaction(store, 'readonly').objectStore(store).get(key);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export function dbGetAll(db, store) {
  return new Promise((resolve, reject) => {
    const req = db.transaction(store, 'readonly').objectStore(store).getAll();
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export function dbPut(db, store, value) {
  return tx(db, store, 'readwrite', (s) => s.put(value));
}

export function dbClear(db, store) {
  return tx(db, store, 'readwrite', (s) => s.clear());
}
