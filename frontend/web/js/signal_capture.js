/**
 * Browser Signal Capture Module
 * 
 * Captures non-PII environmental signals for bot detection
 * Privacy-first: No biometrics, no location, no personal data
 * 
 * IMPORTANT: This code runs in the browser sandbox and is completely
 * client-side. All fingerprinting is ephemeral and used only for
 * passive verification.
 */

class SignalCapture {
  constructor() {
    this.signals = {
      browser: {},
      pointer: {
        movements: [],
        clicks: [],
        hovers: []
      },
      keyboard: {
        pressTimings: [],
        interKeyLatency: []
      },
      scroll: {
        events: [],
        velocities: []
      },
      timing: {
        focusEvents: [],
        blurEvents: [],
        visibilityChanges: []
      },
      network: {
        requestTimings: []
      }
    };

    this.sessionStart = Date.now();
    this.lastActivity = Date.now();
    this.initialized = false;
  }

  /**
   * Initialize all signal collectors
   */
  initialize() {
    if (this.initialized) return;
    
    this.collectBrowserEntropy();
    this.attachPointerListeners();
    this.attachKeyboardListeners();
    this.attachScrollListeners();
    this.attachTimingListeners();
    this.attachNetworkListeners();
    
    this.initialized = true;
    console.log('[SignalCapture] Initialized');
  }

  /**
   * Collect Browser Entropy (15 features)
   * These are stable characteristics that differentiate devices
   */
  collectBrowserEntropy() {
    try {
      const nav = navigator;
      const scr = screen;
      
      this.signals.browser = {
        userAgentHash: this.hashString(nav.userAgent),
        screenResolution: `${scr.width}x${scr.height}`,
        colorDepth: scr.colorDepth,
        pixelRatio: window.devicePixelRatio || 1,
        timezone: new Date().getTimezoneOffset(),
        language: nav.language,
        languages: nav.languages ? nav.languages.join(',') : '',
        platform: nav.platform,
        hardwareConcurrency: nav.hardwareConcurrency || 0,
        deviceMemory: nav.deviceMemory || 0,
        cookieEnabled: nav.cookieEnabled,
        doNotTrack: nav.doNotTrack || 'unknown',
        canvasFingerprint: this.getCanvasFingerprint(),
        webglFingerprint: this.getWebGLFingerprint(),
        audioFingerprint: this.getAudioFingerprint(),
        fontHash: this.getFontFingerprint()
      };
    } catch (error) {
      console.error('[SignalCapture] Error collecting browser entropy:', error);
    }
  }

  /**
   * Attach Pointer Movement Listeners (18 features)
   * Captures mouse/touch movement patterns
   */
  attachPointerListeners() {
    let lastPoint = null;
    let lastTimestamp = null;

    // Mouse movement
    document.addEventListener('mousemove', (e) => {
      this.lastActivity = Date.now();
      const currentPoint = { x: e.clientX, y: e.clientY };
      const timestamp = e.timeStamp;

      if (lastPoint && lastTimestamp) {
        const dx = currentPoint.x - lastPoint.x;
        const dy = currentPoint.y - lastPoint.y;
        const dt = timestamp - lastTimestamp;
        const distance = Math.sqrt(dx * dx + dy * dy);
        const velocity = dt > 0 ? distance / dt : 0;

        this.signals.pointer.movements.push({
          timestamp,
          dx,
          dy,
          dt,
          velocity,
          pressure: e.pressure || 0
        });

        // Keep only last 100 movements to prevent memory bloat
        if (this.signals.pointer.movements.length > 100) {
          this.signals.pointer.movements.shift();
        }
      }

      lastPoint = currentPoint;
      lastTimestamp = timestamp;
    }, { passive: true });

    // Click events
    document.addEventListener('click', (e) => {
      this.lastActivity = Date.now();
      this.signals.pointer.clicks.push({
        timestamp: e.timeStamp,
        x: e.clientX,
        y: e.clientY,
        button: e.button,
        detail: e.detail  // Click count (single, double, etc.)
      });

      if (this.signals.pointer.clicks.length > 50) {
        this.signals.pointer.clicks.shift();
      }
    }, { passive: true });

    // Hover events (mouseover with duration)
    let hoverStart = null;
    document.addEventListener('mouseover', (e) => {
      hoverStart = { timestamp: e.timeStamp, target: e.target.tagName };
    }, { passive: true });

    document.addEventListener('mouseout', (e) => {
      if (hoverStart && hoverStart.target === e.target.tagName) {
        this.signals.pointer.hovers.push({
          element: e.target.tagName,
          duration: e.timeStamp - hoverStart.timestamp
        });

        if (this.signals.pointer.hovers.length > 50) {
          this.signals.pointer.hovers.shift();
        }
      }
    }, { passive: true });

    // Touch events for mobile
    document.addEventListener('touchmove', (e) => {
      this.lastActivity = Date.now();
      if (e.touches.length > 0) {
        const touch = e.touches[0];
        this.signals.pointer.movements.push({
          timestamp: e.timeStamp,
          x: touch.clientX,
          y: touch.clientY,
          type: 'touch',
          force: touch.force || 0
        });
      }
    }, { passive: true });
  }

  /**
   * Attach Keyboard Listeners (12 features)
   * Captures typing dynamics and patterns
   */
  attachKeyboardListeners() {
    let lastKeyTime = null;
    let keyDownTime = {};

    document.addEventListener('keydown', (e) => {
      this.lastActivity = Date.now();
      if (!keyDownTime[e.key]) {
        keyDownTime[e.key] = e.timeStamp;
      }
    }, { passive: true });

    document.addEventListener('keyup', (e) => {
      this.lastActivity = Date.now();
      const keyUpTime = e.timeStamp;
      const keyDown = keyDownTime[e.key];

      if (keyDown) {
        const pressDuration = keyUpTime - keyDown;
        
        // Inter-key latency
        const interKeyLatency = lastKeyTime ? keyDown - lastKeyTime : 0;

        this.signals.keyboard.pressTimings.push({
          timestamp: keyUpTime,
          pressDuration,
          key: e.key.length === 1 ? 'char' : 'special',  // Don't log actual keys (privacy)
          shiftKey: e.shiftKey,
          ctrlKey: e.ctrlKey,
          altKey: e.altKey
        });

        if (interKeyLatency > 0) {
          this.signals.keyboard.interKeyLatency.push(interKeyLatency);
        }

        // Cleanup
        delete keyDownTime[e.key];
        lastKeyTime = keyUpTime;

        // Keep only last 100 entries
        if (this.signals.keyboard.pressTimings.length > 100) {
          this.signals.keyboard.pressTimings.shift();
        }
        if (this.signals.keyboard.interKeyLatency.length > 100) {
          this.signals.keyboard.interKeyLatency.shift();
        }
      }
    }, { passive: true });
  }

  /**
   * Attach Scroll Listeners (8 features)
   * Captures scroll behavior patterns
   */
  attachScrollListeners() {
    let lastScrollY = window.scrollY;
    let lastScrollTime = Date.now();

    window.addEventListener('scroll', (e) => {
      this.lastActivity = Date.now();
      const currentScrollY = window.scrollY;
      const currentTime = Date.now();
      const dy = currentScrollY - lastScrollY;
      const dt = currentTime - lastScrollTime;
      const velocity = dt > 0 ? dy / dt : 0;

      this.signals.scroll.events.push({
        timestamp: currentTime,
        scrollY: currentScrollY,
        deltaY: dy,
        deltaTime: dt
      });

      this.signals.scroll.velocities.push(velocity);

      // Keep only last 50 scroll events
      if (this.signals.scroll.events.length > 50) {
        this.signals.scroll.events.shift();
      }
      if (this.signals.scroll.velocities.length > 50) {
        this.signals.scroll.velocities.shift();
      }

      lastScrollY = currentScrollY;
      lastScrollTime = currentTime;
    }, { passive: true });
  }

  /**
   * Attach Timing Listeners (10 features)
   * Captures focus, blur, and visibility events
   */
  attachTimingListeners() {
    window.addEventListener('focus', () => {
      this.signals.timing.focusEvents.push(Date.now());
    }, { passive: true });

    window.addEventListener('blur', () => {
      this.signals.timing.blurEvents.push(Date.now());
    }, { passive: true });

    document.addEventListener('visibilitychange', () => {
      this.signals.timing.visibilityChanges.push({
        timestamp: Date.now(),
        hidden: document.hidden
      });
    }, { passive: true });
  }

  /**
   * Attach Network Timing Listeners
   * Captures request timing jitter (if accessible)
   */
  attachNetworkListeners() {
    if (window.PerformanceObserver) {
      try {
        const observer = new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) {
            if (entry.entryType === 'resource' || entry.entryType === 'navigation') {
              this.signals.network.requestTimings.push({
                timestamp: Date.now(),
                duration: entry.duration,
                transferSize: entry.transferSize || 0,
                encodedBodySize: entry.encodedBodySize || 0
              });

              // Keep only last 50 entries
              if (this.signals.network.requestTimings.length > 50) {
                this.signals.network.requestTimings.shift();
              }
            }
          }
        });

        observer.observe({ entryTypes: ['resource', 'navigation'] });
      } catch (error) {
        console.warn('[SignalCapture] PerformanceObserver not available:', error);
      }
    }
  }

  /**
   * Get Canvas Fingerprint
   * Stable device identifier based on rendering differences
   */
  getCanvasFingerprint() {
    try {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      
      if (!ctx) return 'unsupported';

      canvas.width = 200;
      canvas.height = 50;

      ctx.textBaseline = 'top';
      ctx.font = '14px Arial';
      ctx.fillStyle = '#f00';
      ctx.fillRect(125, 1, 62, 20);
      ctx.fillStyle = '#069';
      ctx.fillText('BotDetect<🤖>', 2, 15);
      ctx.fillStyle = 'rgba(102, 204, 0, 0.7)';
      ctx.fillText('BotDetect<🤖>', 4, 17);

      return this.hashString(canvas.toDataURL());
    } catch (error) {
      return 'error';
    }
  }

  /**
   * Get WebGL Fingerprint
   */
  getWebGLFingerprint() {
    try {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
      
      if (!gl) return 'unsupported';

      const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
      if (debugInfo) {
        const vendor = gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL);
        const renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
        return this.hashString(`${vendor}~${renderer}`);
      }

      return 'no-debug-info';
    } catch (error) {
      return 'error';
    }
  }

  /**
   * Get Audio Context Fingerprint
   */
  getAudioFingerprint() {
    try {
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const oscillator = audioContext.createOscillator();
      const analyser = audioContext.createAnalyser();
      const gainNode = audioContext.createGain();
      const scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);

      gainNode.gain.value = 0; // Mute
      oscillator.connect(analyser);
      analyser.connect(scriptProcessor);
      scriptProcessor.connect(gainNode);
      gainNode.connect(audioContext.destination);

      oscillator.start(0);
      
      const fingerprint = [
        audioContext.sampleRate,
        audioContext.destination.maxChannelCount,
        audioContext.destination.channelCount
      ].join('_');

      oscillator.stop();
      audioContext.close();

      return this.hashString(fingerprint);
    } catch (error) {
      return 'error';
    }
  }

  /**
   * Get Font Fingerprint
   * Detects available fonts through rendering differences
   */
  getFontFingerprint() {
    try {
      const baseFonts = ['monospace', 'sans-serif', 'serif'];
      const testFonts = [
        'Arial', 'Verdana', 'Times New Roman', 'Courier New',
        'Georgia', 'Palatino', 'Garamond', 'Bookman', 'Comic Sans MS',
        'Trebuchet MS', 'Impact'
      ];

      const testString = 'mmmmmmmmmmlli';
      const testSize = '72px';
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');

      if (!ctx) return 'unsupported';

      const baseMeasurements = {};
      baseFonts.forEach(baseFont => {
        ctx.font = `${testSize} ${baseFont}`;
        baseMeasurements[baseFont] = ctx.measureText(testString).width;
      });

      const detectedFonts = [];
      testFonts.forEach(font => {
        baseFonts.forEach(baseFont => {
          ctx.font = `${testSize} ${font}, ${baseFont}`;
          const measurement = ctx.measureText(testString).width;
          if (measurement !== baseMeasurements[baseFont]) {
            detectedFonts.push(font);
          }
        });
      });

      return this.hashString(detectedFonts.join(','));
    } catch (error) {
      return 'error';
    }
  }

  /**
   * Hash string to numeric value (simple hash for privacy)
   */
  hashString(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash);
  }

  /**
   * Aggregate signals and compute statistical features
   * Returns privacy-safe aggregate features
   */
  aggregateSignals() {
    const now = Date.now();
    const sessionDuration = (now - this.sessionStart) / 1000; // seconds
    const idleTime = (now - this.lastActivity) / 1000; // seconds

    return {
      // Browser entropy (stable features)
      browser: this.signals.browser,

      // Pointer dynamics (statistical aggregates)
      pointer: {
        movementCount: this.signals.pointer.movements.length,
        clickCount: this.signals.pointer.clicks.length,
        hoverCount: this.signals.pointer.hovers.length,
        
        // Velocity statistics
        velocityStats: this.computeStats(
          this.signals.pointer.movements.map(m => m.velocity)
        ),
        
        // Movement smoothness (variance in velocity)
        smoothness: this.computeVariance(
          this.signals.pointer.movements.map(m => m.velocity)
        ),
        
        // Direction changes
        directionChanges: this.countDirectionChanges(this.signals.pointer.movements),
        
        // Pause detection
        pauseCount: this.countPauses(this.signals.pointer.movements),
        
        // Average hover duration
        avgHoverDuration: this.computeMean(
          this.signals.pointer.hovers.map(h => h.duration)
        )
      },

      // Keyboard dynamics
      keyboard: {
        keyPressCount: this.signals.keyboard.pressTimings.length,
        
        // Press duration statistics
        pressDurationStats: this.computeStats(
          this.signals.keyboard.pressTimings.map(k => k.pressDuration)
        ),
        
        // Inter-key latency statistics
        interKeyLatencyStats: this.computeStats(this.signals.keyboard.interKeyLatency),
        
        // Typing rhythm (variance in timing)
        typingRhythm: this.computeVariance(this.signals.keyboard.interKeyLatency),
        
        // Modifier key usage rate
        modifierKeyRate: this.computeModifierKeyRate()
      },

      // Scroll behavior
      scroll: {
        scrollCount: this.signals.scroll.events.length,
        
        // Velocity statistics
        velocityStats: this.computeStats(this.signals.scroll.velocities),
        
        // Smoothness
        smoothness: this.computeVariance(this.signals.scroll.velocities),
        
        // Direction changes
        directionChanges: this.countScrollDirectionChanges()
      },

      // Timing and interaction patterns
      timing: {
        sessionDuration,
        idleTime,
        focusEventCount: this.signals.timing.focusEvents.length,
        blurEventCount: this.signals.timing.blurEvents.length,
        visibilityChangeCount: this.signals.timing.visibilityChanges.length,
        
        // Activity frequency
        activityFrequency: (
          this.signals.pointer.movements.length +
          this.signals.keyboard.pressTimings.length +
          this.signals.scroll.events.length
        ) / sessionDuration
      },

      // Network characteristics
      network: {
        requestCount: this.signals.network.requestTimings.length,
        
        // Request timing statistics
        durationStats: this.computeStats(
          this.signals.network.requestTimings.map(r => r.duration)
        ),
        
        // Transfer size statistics
        transferSizeStats: this.computeStats(
          this.signals.network.requestTimings.map(r => r.transferSize)
        )
      },

      // Metadata
      metadata: {
        timestamp: now,
        sessionStart: this.sessionStart,
        featureVersion: '1.0.0'
      }
    };
  }

  /**
   * Statistical helper functions
   */
  computeStats(values) {
    if (!values || values.length === 0) {
      return { mean: 0, std: 0, min: 0, max: 0, median: 0 };
    }

    const sorted = [...values].sort((a, b) => a - b);
    const mean = this.computeMean(values);
    const std = Math.sqrt(this.computeVariance(values));

    return {
      mean,
      std,
      min: sorted[0],
      max: sorted[sorted.length - 1],
      median: sorted[Math.floor(sorted.length / 2)],
      p25: sorted[Math.floor(sorted.length * 0.25)],
      p75: sorted[Math.floor(sorted.length * 0.75)]
    };
  }

  computeMean(values) {
    if (!values || values.length === 0) return 0;
    return values.reduce((sum, val) => sum + val, 0) / values.length;
  }

  computeVariance(values) {
    if (!values || values.length < 2) return 0;
    const mean = this.computeMean(values);
    const squaredDiffs = values.map(val => Math.pow(val - mean, 2));
    return this.computeMean(squaredDiffs);
  }

  countDirectionChanges(movements) {
    if (movements.length < 3) return 0;
    
    let changes = 0;
    for (let i = 2; i < movements.length; i++) {
      const prev = movements[i - 1];
      const curr = movements[i];
      
      // Check if direction changed
      const prevAngle = Math.atan2(prev.dy, prev.dx);
      const currAngle = Math.atan2(curr.dy, curr.dx);
      const angleDiff = Math.abs(currAngle - prevAngle);
      
      if (angleDiff > Math.PI / 4) { // 45 degrees threshold
        changes++;
      }
    }
    
    return changes;
  }

  countPauses(movements) {
    if (movements.length < 2) return 0;
    
    let pauses = 0;
    const pauseThreshold = 100; // ms
    
    for (let i = 1; i < movements.length; i++) {
      if (movements[i].dt > pauseThreshold) {
        pauses++;
      }
    }
    
    return pauses;
  }

  countScrollDirectionChanges() {
    const velocities = this.signals.scroll.velocities;
    if (velocities.length < 2) return 0;
    
    let changes = 0;
    for (let i = 1; i < velocities.length; i++) {
      if (Math.sign(velocities[i]) !== Math.sign(velocities[i - 1])) {
        changes++;
      }
    }
    
    return changes;
  }

  computeModifierKeyRate() {
    const timings = this.signals.keyboard.pressTimings;
    if (timings.length === 0) return 0;
    
    const modifierCount = timings.filter(
      t => t.shiftKey || t.ctrlKey || t.altKey
    ).length;
    
    return modifierCount / timings.length;
  }

  /**
   * Reset signals (call after successful submission)
   */
  reset() {
    this.signals.pointer.movements = [];
    this.signals.pointer.clicks = [];
    this.signals.pointer.hovers = [];
    this.signals.keyboard.pressTimings = [];
    this.signals.keyboard.interKeyLatency = [];
    this.signals.scroll.events = [];
    this.signals.scroll.velocities = [];
    this.signals.timing.focusEvents = [];
    this.signals.timing.blurEvents = [];
    this.signals.timing.visibilityChanges = [];
    this.signals.network.requestTimings = [];
    
    this.sessionStart = Date.now();
    this.lastActivity = Date.now();
  }
}

// Export for use in Flutter via JS interop
window.SignalCapture = SignalCapture;

// Auto-initialize if not in test environment
if (typeof window !== 'undefined' && !window.__SIGNAL_CAPTURE_TEST__) {
  window.botDetectionSignals = new SignalCapture();
  window.botDetectionSignals.initialize();
  console.log('[SignalCapture] Auto-initialized');
}
