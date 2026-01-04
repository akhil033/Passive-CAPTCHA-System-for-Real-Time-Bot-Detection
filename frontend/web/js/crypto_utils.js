/**
 * Cryptographic Utilities for Payload Signing
 * 
 * Implements HMAC-SHA256 signing for request authentication
 * Prevents replay attacks through nonce and timestamp validation
 */

class CryptoUtils {
  constructor() {
    this.sessionKey = null;
    this.sessionId = null;
  }

  /**
   * Initialize session with server-provided key
   * @param {string} sessionId - Unique session identifier
   * @param {string} sessionKey - Base64-encoded HMAC key
   */
  async initializeSession(sessionId, sessionKey) {
    this.sessionId = sessionId;
    this.sessionKey = await this.importKey(sessionKey);
    console.log('[CryptoUtils] Session initialized:', sessionId);
  }

  /**
   * Import base64-encoded key for HMAC operations
   */
  async importKey(base64Key) {
    try {
      const keyData = this.base64ToArrayBuffer(base64Key);
      return await window.crypto.subtle.importKey(
        'raw',
        keyData,
        { name: 'HMAC', hash: 'SHA-256' },
        false,
        ['sign', 'verify']
      );
    } catch (error) {
      console.error('[CryptoUtils] Key import failed:', error);
      throw new Error('Failed to import session key');
    }
  }

  /**
   * Sign a payload with HMAC-SHA256
   * @param {object} payload - Data to sign
   * @returns {object} - Signed payload with timestamp, nonce, and signature
   */
  async signPayload(payload) {
    if (!this.sessionKey || !this.sessionId) {
      throw new Error('Session not initialized. Call initializeSession first.');
    }

    const timestamp = Date.now();
    const nonce = this.generateNonce();

    // Create canonical string to sign
    const canonicalString = this.createCanonicalString(
      this.sessionId,
      timestamp,
      nonce,
      payload
    );

    // Compute HMAC signature
    const signature = await this.computeHMAC(canonicalString);

    return {
      sessionId: this.sessionId,
      timestamp,
      nonce,
      payload: this.compressPayload(payload),
      signature
    };
  }

  /**
   * Create canonical string for signing
   * Format: sessionId|timestamp|nonce|payloadHash
   */
  createCanonicalString(sessionId, timestamp, nonce, payload) {
    const payloadJson = JSON.stringify(payload);
    const payloadHash = this.hashString(payloadJson);
    return `${sessionId}|${timestamp}|${nonce}|${payloadHash}`;
  }

  /**
   * Compute HMAC-SHA256 signature
   */
  async computeHMAC(data) {
    try {
      const encoder = new TextEncoder();
      const dataBuffer = encoder.encode(data);
      const signatureBuffer = await window.crypto.subtle.sign(
        'HMAC',
        this.sessionKey,
        dataBuffer
      );
      return this.arrayBufferToBase64(signatureBuffer);
    } catch (error) {
      console.error('[CryptoUtils] HMAC computation failed:', error);
      throw new Error('Failed to compute signature');
    }
  }

  /**
   * Generate cryptographically secure nonce
   * @returns {string} - 32-byte base64-encoded random nonce
   */
  generateNonce() {
    const nonceArray = new Uint8Array(32);
    window.crypto.getRandomValues(nonceArray);
    return this.arrayBufferToBase64(nonceArray);
  }

  /**
   * Hash string using SHA-256
   */
  async hashString(str) {
    const encoder = new TextEncoder();
    const data = encoder.encode(str);
    const hashBuffer = await window.crypto.subtle.digest('SHA-256', data);
    return this.arrayBufferToBase64(hashBuffer);
  }

  /**
   * Compress payload using gzip (if supported)
   * Falls back to base64 encoding if compression not available
   */
  compressPayload(payload) {
    const jsonString = JSON.stringify(payload);
    
    // Check if CompressionStream is available (modern browsers)
    if (typeof CompressionStream !== 'undefined') {
      // Return promise that will be resolved by caller
      return this.gzipCompress(jsonString);
    } else {
      // Fallback: base64 encode without compression
      return this.stringToBase64(jsonString);
    }
  }

  /**
   * Gzip compression using CompressionStream API
   */
  async gzipCompress(str) {
    try {
      const encoder = new TextEncoder();
      const data = encoder.encode(str);
      
      const stream = new ReadableStream({
        start(controller) {
          controller.enqueue(data);
          controller.close();
        }
      });

      const compressedStream = stream.pipeThrough(new CompressionStream('gzip'));
      const chunks = [];
      
      const reader = compressedStream.getReader();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
      }

      // Concatenate chunks
      const totalLength = chunks.reduce((acc, chunk) => acc + chunk.length, 0);
      const compressed = new Uint8Array(totalLength);
      let offset = 0;
      for (const chunk of chunks) {
        compressed.set(chunk, offset);
        offset += chunk.length;
      }

      return this.arrayBufferToBase64(compressed);
    } catch (error) {
      console.warn('[CryptoUtils] Gzip compression failed, using base64:', error);
      return this.stringToBase64(str);
    }
  }

  /**
   * Verify timestamp is within acceptable window
   * @param {number} timestamp - Timestamp to verify
   * @param {number} windowMs - Acceptable time window in milliseconds (default 5 minutes)
   */
  isTimestampValid(timestamp, windowMs = 300000) {
    const now = Date.now();
    const diff = Math.abs(now - timestamp);
    return diff <= windowMs;
  }

  /**
   * Convert ArrayBuffer to base64 string
   */
  arrayBufferToBase64(buffer) {
    const bytes = new Uint8Array(buffer);
    let binary = '';
    for (let i = 0; i < bytes.byteLength; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return window.btoa(binary);
  }

  /**
   * Convert base64 string to ArrayBuffer
   */
  base64ToArrayBuffer(base64) {
    const binary = window.atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    return bytes.buffer;
  }

  /**
   * Convert string to base64
   */
  stringToBase64(str) {
    return window.btoa(unescape(encodeURIComponent(str)));
  }

  /**
   * Convert base64 to string
   */
  base64ToString(base64) {
    return decodeURIComponent(escape(window.atob(base64)));
  }

  /**
   * Generate device fingerprint hash
   * Combines multiple entropy sources for a stable device identifier
   */
  async generateDeviceFingerprint(browserEntropy) {
    const fingerprintData = {
      userAgentHash: browserEntropy.userAgentHash,
      screenResolution: browserEntropy.screenResolution,
      colorDepth: browserEntropy.colorDepth,
      timezone: browserEntropy.timezone,
      language: browserEntropy.language,
      platform: browserEntropy.platform,
      canvasFingerprint: browserEntropy.canvasFingerprint,
      webglFingerprint: browserEntropy.webglFingerprint,
      audioFingerprint: browserEntropy.audioFingerprint,
      fontHash: browserEntropy.fontHash
    };

    const fingerprintString = JSON.stringify(fingerprintData);
    return await this.hashString(fingerprintString);
  }

  /**
   * Rotate session key (call periodically or after suspicious activity)
   */
  async rotateSessionKey(newSessionKey) {
    if (!this.sessionId) {
      throw new Error('No active session to rotate');
    }

    console.log('[CryptoUtils] Rotating session key');
    this.sessionKey = await this.importKey(newSessionKey);
  }

  /**
   * Clear session data
   */
  clearSession() {
    this.sessionKey = null;
    this.sessionId = null;
    console.log('[CryptoUtils] Session cleared');
  }
}

// Export for use in Flutter via JS interop
window.CryptoUtils = CryptoUtils;

// Auto-initialize if not in test environment
if (typeof window !== 'undefined' && !window.__CRYPTO_UTILS_TEST__) {
  window.botDetectionCrypto = new CryptoUtils();
  console.log('[CryptoUtils] Initialized');
}
