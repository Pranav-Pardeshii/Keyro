/**
 * Keystroke Dynamics High-Precision Timestamp Logger & Feature Extractor
 * Target Passphrase: .tie5Roanl + Enter (11 keys total)
 * Features: 31 dimensions (11 Hold, 10 Down-Down, 10 Up-Down)
 */

class KeystrokeRecorder {
    constructor(options = {}) {
        this.targetPassphrase = ".tie5Roanl";
        this.expectedKeys = [".", "t", "i", "e", "5", "R", "o", "a", "n", "l", "Enter"];
        this.onProgress = options.onProgress || (() => {});
        this.onComplete = options.onComplete || (() => {});
        this.onError = options.onError || (() => {});
        this.onKeyTiming = options.onKeyTiming || (() => {});

        this.downEvents = [];
        this.activeDownKeys = new Map();
        this.isComplete = false;
    }

    reset() {
        this.downEvents = [];
        this.activeDownKeys.clear();
        this.isComplete = false;
        this.onProgress({
            typedIndex: -1,
            nextExpectedIndex: 0,
            targetIndex: 0,
            currentIndex: -1,
            totalKeys: this.expectedKeys.length
        });
    }

    handleKeyDown(event) {
        if (this.isComplete) return;

        const key = event.key;

        // Allow Backspace or Escape to reset current typing attempt
        if (key === "Escape" || key === "Backspace") {
            this.reset();
            this.onError("Attempt reset. Please re-type '.tie5Roanl' and press Return.");
            return;
        }

        // Ignore modifier-only presses like isolated Shift, Control, Alt, Meta, CapsLock, Tab
        if (["Shift", "Control", "Alt", "Meta", "CapsLock", "Tab"].includes(key)) {
            return;
        }

        // Prevent default browser behavior on Enter (e.g. form submission or sound)
        if (key === "Enter") {
            event.preventDefault();
        }

        const expectedIndex = this.downEvents.length;
        if (expectedIndex >= this.expectedKeys.length) {
            return;
        }

        const expectedKey = this.expectedKeys[expectedIndex];
        const isMatch = (expectedKey.toLowerCase() === key.toLowerCase());

        if (!isMatch) {
            this.reset();
            const expectedLabel = expectedKey === "Enter" ? "Return" : expectedKey;
            const keyLabel = key === "Enter" ? "Return" : key;
            this.onError(`Typo! Expected '${expectedLabel}' at position ${expectedIndex + 1}, but got '${keyLabel}'. Attempt reset.`);
            return;
        }

        // High precision timestamp in seconds
        const downTime = performance.now() / 1000.0;
        const eventData = {
            index: expectedIndex,
            key: expectedKey,
            typedKey: key,
            downTime: downTime,
            upTime: null
        };

        this.downEvents.push(eventData);
        this.activeDownKeys.set(expectedIndex, eventData);

        const nextTargetIndex = expectedIndex + 1;
        this.onProgress({
            typedIndex: expectedIndex,
            nextExpectedIndex: nextTargetIndex,
            targetIndex: nextTargetIndex,
            currentIndex: expectedIndex,
            totalKeys: this.expectedKeys.length,
            key: key,
            expectedKey: expectedKey
        });
    }

    handleKeyUp(event) {
        if (this.isComplete) return;

        const key = event.key;
        if (["Shift", "Control", "Alt", "Meta", "CapsLock", "Tab"].includes(key)) {
            return;
        }

        if (key === "Enter") {
            event.preventDefault();
        }

        // Find matching pending down key
        let matchedIndex = null;
        for (const [idx, data] of this.activeDownKeys.entries()) {
            if (data.key.toLowerCase() === key.toLowerCase() && data.upTime === null) {
                matchedIndex = idx;
                break;
            }
        }

        if (matchedIndex === null) {
            return;
        }

        const upTime = performance.now() / 1000.0;
        const eventData = this.activeDownKeys.get(matchedIndex);
        eventData.upTime = upTime;
        this.activeDownKeys.delete(matchedIndex);

        const holdMs = (upTime - eventData.downTime) * 1000.0;
        this.onKeyTiming({
            keyIndex: matchedIndex,
            key: eventData.key,
            holdMs: Math.round(holdMs)
        });

        // Check if all 11 keys have been completed
        if (this.downEvents.length === 11) {
            // Safety check: ensure all 11 keys have valid upTime
            for (let i = 0; i < 11; i++) {
                if (this.downEvents[i].upTime === null) {
                    if (i < 10) {
                        this.downEvents[i].upTime = Math.min(
                            this.downEvents[i + 1].downTime,
                            this.downEvents[i].downTime + 0.120
                        );
                    } else {
                        this.downEvents[i].upTime = upTime;
                    }
                }
            }

            const allHaveUp = this.downEvents.every(ev => ev && ev.upTime !== null);
            if (allHaveUp) {
                this.isComplete = true;
                const features = this.computeFeatures();
                const rawCopy = this.downEvents.map(e => ({ ...e }));
                this.onComplete({
                    rawEvents: rawCopy,
                    features: features
                });
            }
        }
    }

    computeFeatures() {
        if (this.downEvents.length !== 11) {
            throw new Error(`Expected 11 key events, found ${this.downEvents.length}`);
        }

        const downTimes = this.downEvents.map(e => e.downTime);
        const upTimes = this.downEvents.map(e => e.upTime);

        const features = [];
        // 10 key transitions
        for (let m = 0; m < 10; m++) {
            const h = upTimes[m] - downTimes[m];             // H.km
            const dd = downTimes[m + 1] - downTimes[m];     // DD.km.k_{m+1}
            const ud = downTimes[m + 1] - upTimes[m];       // UD.km.k_{m+1}

            features.push(Number(h.toFixed(6)));
            features.push(Number(dd.toFixed(6)));
            features.push(Number(ud.toFixed(6)));
        }

        // Final Return hold time
        const hReturn = upTimes[10] - downTimes[10];
        features.push(Number(hReturn.toFixed(6)));

        return features;
    }
}

// Export to window for browser use
window.KeystrokeRecorder = KeystrokeRecorder;
