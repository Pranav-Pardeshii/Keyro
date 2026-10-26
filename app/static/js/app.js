/**
 * Keyro Frontend Application Controller — "The Product Performs" Edition
 * Handles Hero live sensor, enrollment workflow, verification gateway,
 * real-time threshold slider calibration, and adversarial defense arena.
 */

document.addEventListener("DOMContentLoaded", () => {
    // Autofill Purge: Guarantee all typing & text inputs boot completely empty on refresh
    function purgeAllInputFields() {
        document.querySelectorAll(".sensor-input, .text-input").forEach(el => {
            el.value = "";
            el.defaultValue = "";
        });
    }
    purgeAllInputFields();
    window.addEventListener("load", purgeAllInputFields);
    window.addEventListener("pageshow", purgeAllInputFields);
    setTimeout(purgeAllInputFields, 50);
    setTimeout(purgeAllInputFields, 200);
    setTimeout(purgeAllInputFields, 500);

    // Keycap Highlighting: The key TO BE PRESSED is highlighted with cyan (.active)
    // Keys already pressed enter calm completed green (.typed)
    function highlightNextKey(slots, targetIndex) {
        if (!slots || slots.length === 0) return;
        const target = (typeof targetIndex === "number") ? targetIndex : 0;
        slots.forEach((slot, idx) => {
            if (idx < target) {
                slot.classList.remove("active");
                slot.classList.add("typed");
            } else if (idx === target) {
                slot.classList.remove("typed");
                slot.classList.add("active");
            } else {
                slot.classList.remove("active", "typed");
            }
        });
    }

    // State
    const state = {
        users: [],
        activeUser: null,
        enrollment: {
            username: "",
            modelType: "One-Class SVM",
            samples: [],
            targetReps: 20
        },
        tradeoffs: []
    };

    // DOM Elements - Navigation Tabs
    const tabButtons = document.querySelectorAll(".tab-button, .tab-btn");
    const tabPanes = document.querySelectorAll(".workspace-pane, .tab-pane");

    function switchTab(targetId, shouldScroll = false) {
        tabButtons.forEach(b => {
            const matches = b.getAttribute("data-tab") === targetId;
            b.classList.toggle("active", matches);
            b.setAttribute("aria-selected", matches ? "true" : "false");
        });

        tabPanes.forEach(p => {
            const matches = p.id === targetId;
            p.classList.toggle("active", matches);
        });

        if (targetId === "tab-verify" || targetId === "tab-impersonate") {
            loadUsers();
        }

        if (shouldScroll) {
            const wrapper = document.querySelector(".workspaces-wrapper");
            if (wrapper) {
                wrapper.scrollIntoView({ behavior: "smooth", block: "start" });
            }
        }
    }

    tabButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetId = btn.getAttribute("data-tab");
            switchTab(targetId, true);
        });
    });

    // Hero quick jump actions
    const heroBtnEnroll = document.getElementById("hero-btn-enroll");
    if (heroBtnEnroll) {
        heroBtnEnroll.addEventListener("click", () => {
            switchTab("tab-enroll", true);
            const userInput = document.getElementById("enroll-username");
            if (userInput) userInput.focus();
        });
    }

    const heroBtnVerify = document.getElementById("hero-btn-verify");
    if (heroBtnVerify) {
        heroBtnVerify.addEventListener("click", () => {
            switchTab("tab-verify", true);
        });
    }

    // -------------------------------------------------------------------------
    // API Helper Functions
    // -------------------------------------------------------------------------
    async function apiRequest(endpoint, method = "GET", body = null) {
        const options = {
            method,
            headers: { "Content-Type": "application/json" }
        };
        if (body) {
            options.body = JSON.stringify(body);
        }
        const res = await fetch(endpoint, options);
        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || `HTTP Error ${res.status}`);
        }
        return await res.json();
    }

    async function loadUsers() {
        try {
            const users = await apiRequest("/users");
            state.users = users;

            const verifySelect = document.getElementById("verify-user-select");
            const impSelect = document.getElementById("impersonate-target-user");

            if (!verifySelect || !impSelect) return;

            const currentVerifyVal = verifySelect.value;
            const currentImpVal = impSelect.value;

            verifySelect.innerHTML = '<option value="">Choose User Profile</option>';
            impSelect.innerHTML = '<option value="">Choose Victim Profile</option>';

            users.forEach(u => {
                const opt1 = document.createElement("option");
                opt1.value = u.username;
                opt1.textContent = `${u.username} (${u.model_type}, Thresh: ${u.default_threshold.toFixed(4)})`;
                verifySelect.appendChild(opt1);

                const opt2 = document.createElement("option");
                opt2.value = u.username;
                opt2.textContent = `${u.username} (${u.model_type})`;
                impSelect.appendChild(opt2);
            });

            // Restore selection or select first
            if (currentVerifyVal && users.some(u => u.username === currentVerifyVal)) {
                verifySelect.value = currentVerifyVal;
            } else if (users.length > 0) {
                verifySelect.value = users[0].username;
                onVerifyUserChanged(users[0].username);
            }

            if (currentImpVal && users.some(u => u.username === currentImpVal)) {
                impSelect.value = currentImpVal;
            } else if (users.length > 0) {
                impSelect.value = users[0].username;
            }
        } catch (e) {
            console.error("Failed to load users:", e);
        }
    }

    // -------------------------------------------------------------------------
    // HERO SENSOR: "The Product Performs" Live Interactive Cadence Instrument
    // -------------------------------------------------------------------------
    const heroInput = document.getElementById("hero-typing-input");
    const heroSlots = document.querySelectorAll("#hero-char-slots .char-slot");
    const heroTimeline = document.getElementById("hero-cadence-timeline");
    const heroFeedback = document.getElementById("hero-feedback");
    const heroMetricHold = document.getElementById("hero-metric-hold");
    const heroMetricFlight = document.getElementById("hero-metric-flight");
    const btnHeroDemoPulse = document.getElementById("btn-hero-demo-pulse");

    let heroHoldTimes = [];

    function resetHeroSlots() {
        highlightNextKey(heroSlots, 0);
    }

    if (heroInput && heroSlots.length > 0) {
        const heroRecorder = new KeystrokeRecorder({
            onProgress: (data) => {
                highlightNextKey(heroSlots, data.targetIndex);
            },
            onKeyTiming: (data) => {
                heroHoldTimes.push(data.holdMs);
                const avgHold = (heroHoldTimes.reduce((a, b) => a + b, 0) / heroHoldTimes.length).toFixed(1);
                if (heroMetricHold) heroMetricHold.textContent = `${avgHold} ms`;

                if (heroTimeline) {
                    const bar = document.createElement("div");
                    bar.className = "cadence-bar";
                    const h = Math.min(48, Math.max(6, data.holdMs / 4.5));
                    bar.style.height = `${h}px`;
                    bar.title = `${data.key}: ${data.holdMs}ms`;
                    heroTimeline.appendChild(bar);
                    if (heroTimeline.children.length > 22) {
                        heroTimeline.removeChild(heroTimeline.firstChild);
                    }
                }
            },
            onError: (msg) => {
                if (heroFeedback) {
                    heroFeedback.textContent = msg;
                    heroFeedback.style.color = "var(--danger)";
                }
                heroInput.value = "";
                resetHeroSlots();
                heroRecorder.reset();
                setTimeout(() => {
                    if (heroFeedback) {
                        heroFeedback.textContent = "Click sensor and type .tie5Roanl to observe neuromuscular latency extraction in real-time.";
                        heroFeedback.style.color = "var(--muted)";
                    }
                }, 3000);
            },
            onComplete: (result) => {
                highlightNextKey(heroSlots, 11);
                if (heroFeedback) {
                    heroFeedback.textContent = "Neuromuscular latency signature captured (31 dimensions).";
                    heroFeedback.style.color = "var(--accent)";
                }

                // Flight calculation
                if (result.features && result.features.length >= 21) {
                    const ddTimes = result.features.slice(11, 21);
                    const avgDd = (ddTimes.reduce((a, b) => a + b, 0) / ddTimes.length * 1000).toFixed(1);
                    if (heroMetricFlight) heroMetricFlight.textContent = `${avgDd} ms`;
                }

                setTimeout(() => {
                    heroInput.value = "";
                    resetHeroSlots();
                    heroRecorder.reset();
                }, 1400);
            }
        });

        heroInput.addEventListener("keydown", (e) => heroRecorder.handleKeyDown(e));
        heroInput.addEventListener("keyup", (e) => heroRecorder.handleKeyUp(e));
    }

    if (btnHeroDemoPulse && heroSlots.length > 0) {
        btnHeroDemoPulse.addEventListener("click", () => {
            btnHeroDemoPulse.disabled = true;
            resetHeroSlots();
            if (heroTimeline) heroTimeline.innerHTML = "";
            heroHoldTimes = [];

            const sampleCadence = [
                { key: ".", hold: 84, flight: 110 },
                { key: "t", hold: 92, flight: 125 },
                { key: "i", hold: 78, flight: 98 },
                { key: "e", hold: 88, flight: 105 },
                { key: "5", hold: 114, flight: 140 },
                { key: "R", hold: 102, flight: 130 },
                { key: "o", hold: 86, flight: 95 },
                { key: "a", hold: 79, flight: 112 },
                { key: "n", hold: 85, flight: 102 },
                { key: "l", hold: 90, flight: 120 },
                { key: "Enter", hold: 108, flight: 150 }
            ];

            let delay = 0;
            highlightNextKey(heroSlots, 0);

            sampleCadence.forEach((item, idx) => {
                setTimeout(() => {
                    // Key idx finished, highlight next key (idx + 1)
                    highlightNextKey(heroSlots, idx + 1);

                    heroHoldTimes.push(item.hold);
                    const avgHold = (heroHoldTimes.reduce((a, b) => a + b, 0) / heroHoldTimes.length).toFixed(1);
                    if (heroMetricHold) heroMetricHold.textContent = `${avgHold} ms`;
                    if (heroMetricFlight) heroMetricFlight.textContent = `${item.flight} ms`;

                    if (heroTimeline) {
                        const bar = document.createElement("div");
                        bar.className = "cadence-bar";
                        bar.style.height = `${Math.min(48, item.hold / 4.2)}px`;
                        heroTimeline.appendChild(bar);
                    }

                    if (idx === sampleCadence.length - 1) {
                        if (heroFeedback) {
                            heroFeedback.textContent = "Simulated genuine cadence completed: 31 neuromuscular features logged.";
                            heroFeedback.style.color = "var(--success)";
                        }
                        btnHeroDemoPulse.disabled = false;
                        setTimeout(() => resetHeroSlots(), 2000);
                    }
                }, delay);
                delay += item.flight;
            });
        });
    }

    // -------------------------------------------------------------------------
    // WORKSPACE 1: ENROLLMENT LOGIC (20 Repetitions)
    // -------------------------------------------------------------------------
    const enrollInput = document.getElementById("enroll-typing-input");
    const enrollSlots = document.querySelectorAll("#enroll-char-slots .char-slot");
    const enrollProgressText = document.getElementById("enroll-progress-text");
    const enrollProgressBar = document.getElementById("enroll-progress-bar");
    const enrollFeedback = document.getElementById("enroll-feedback");
    const enrollCard = document.getElementById("enroll-passphrase-card");
    const enrollTimeline = document.getElementById("enroll-cadence-timeline");
    const enrollSuccessBanner = document.getElementById("enroll-success-banner");
    const enrollSuccessDetails = document.getElementById("enroll-success-details");
    const btnResetEnrollment = document.getElementById("btn-reset-enrollment");
    const btnGotoVerify = document.getElementById("btn-goto-verify");

    function resetEnrollSlots() {
        highlightNextKey(enrollSlots, 0);
    }

    function updateEnrollProgress() {
        const count = state.enrollment.samples.length;
        const total = state.enrollment.targetReps;
        const pct = Math.round((count / total) * 100);

        if (enrollProgressText) enrollProgressText.textContent = `${count} / ${total} Repetitions (${pct}%)`;
        if (enrollProgressBar) enrollProgressBar.style.width = `${pct}%`;

        const badge = document.getElementById("enroll-status-badge");
        if (badge) {
            if (count === 0) {
                badge.textContent = "Ready to Enroll";
                badge.className = "badge-status";
            } else if (count < total) {
                badge.textContent = `Enrolling (${count}/${total})`;
                badge.className = "badge-status";
            } else {
                badge.textContent = "Enrolled";
                badge.className = "badge-status";
            }
        }
    }

    if (enrollInput && enrollSlots.length > 0) {
        const enrollRecorder = new KeystrokeRecorder({
            onProgress: (data) => {
                highlightNextKey(enrollSlots, data.targetIndex);
            },
            onKeyTiming: (data) => {
                if (enrollTimeline) {
                    const bar = document.createElement("div");
                    bar.className = "cadence-bar";
                    const h = Math.min(48, Math.max(6, data.holdMs / 5));
                    bar.style.height = `${h}px`;
                    bar.title = `${data.key}: ${data.holdMs}ms`;
                    enrollTimeline.appendChild(bar);
                    if (enrollTimeline.children.length > 22) {
                        enrollTimeline.removeChild(enrollTimeline.firstChild);
                    }
                }
            },
            onError: (msg) => {
                if (enrollFeedback) {
                    enrollFeedback.textContent = msg;
                    enrollFeedback.style.color = "var(--danger)";
                }
                enrollInput.value = "";
                resetEnrollSlots();
                enrollRecorder.reset();
                setTimeout(() => {
                    if (enrollFeedback) {
                        enrollFeedback.textContent = "Strike each key naturally. Consistency of timing is evaluated, not speed.";
                        enrollFeedback.style.color = "var(--muted)";
                    }
                }, 3000);
            },
            onComplete: async (result) => {
                const username = document.getElementById("enroll-username").value.trim();
                if (!username) {
                    if (enrollFeedback) {
                        enrollFeedback.textContent = "Please enter a Username before typing enrollment samples.";
                        enrollFeedback.style.color = "var(--danger)";
                    }
                    enrollInput.value = "";
                    resetEnrollSlots();
                    enrollRecorder.reset();
                    const userInput = document.getElementById("enroll-username");
                    if (userInput) userInput.focus();
                    return;
                }

                state.enrollment.samples.push(result.features);
                enrollInput.value = "";
                resetEnrollSlots();
                enrollRecorder.reset();
                updateEnrollProgress();

                const count = state.enrollment.samples.length;
                const total = state.enrollment.targetReps;

                if (count < total) {
                    if (enrollFeedback) {
                        enrollFeedback.textContent = `Sample ${count} of ${total} captured. Continue typing at your natural rhythm.`;
                        enrollFeedback.style.color = "var(--accent)";
                    }
                    enrollInput.focus();
                } else {
                    // All 20 samples captured -> Train User Model
                    if (enrollFeedback) {
                        enrollFeedback.textContent = "Training personalized anomaly detector...";
                        enrollFeedback.style.color = "var(--warning)";
                    }
                    enrollInput.disabled = true;

                    try {
                        const modelType = document.getElementById("enroll-model-select").value;
                        const res = await apiRequest("/enroll", "POST", {
                            username: username,
                            samples: state.enrollment.samples,
                            model_type: modelType
                        });

                        if (enrollFeedback) {
                            enrollFeedback.textContent = "Profile enrolled successfully.";
                            enrollFeedback.style.color = "var(--success)";
                        }
                        if (enrollSuccessBanner) {
                            enrollSuccessBanner.style.display = "flex";
                        }
                        if (enrollSuccessDetails) {
                            enrollSuccessDetails.innerHTML = `Model <strong>${res.model_type}</strong> fitted with 20 genuine repetitions. Calibrated baseline threshold: <code>${res.threshold.toFixed(4)}</code>.`;
                        }
                        await loadUsers();
                    } catch (err) {
                        if (enrollFeedback) {
                            enrollFeedback.textContent = `Enrollment error: ${err.message}`;
                            enrollFeedback.style.color = "var(--danger)";
                        }
                        enrollInput.disabled = false;
                        enrollRecorder.reset();
                    }
                }
            }
        });

        enrollInput.addEventListener("keydown", (e) => enrollRecorder.handleKeyDown(e));
        enrollInput.addEventListener("keyup", (e) => enrollRecorder.handleKeyUp(e));
        enrollInput.addEventListener("focus", () => {
            if (enrollCard) enrollCard.classList.add("focused");
            if (enrollRecorder.downEvents.length === 0) {
                enrollInput.value = "";
            }
        });
        enrollInput.addEventListener("blur", () => enrollCard && enrollCard.classList.remove("focused"));
    }

    if (btnResetEnrollment) {
        btnResetEnrollment.addEventListener("click", () => {
            state.enrollment.samples = [];
            if (enrollInput) {
                enrollInput.value = "";
                enrollInput.disabled = false;
                enrollInput.focus();
            }
            const activeRecorder = document.getElementById("enroll-typing-input");
            if (activeRecorder && window.enrollRecorderRef) {
                window.enrollRecorderRef.reset();
            }
            if (typeof enrollRecorder !== "undefined" && enrollRecorder) {
                enrollRecorder.reset();
            }
            if (enrollSuccessBanner) enrollSuccessBanner.style.display = "none";
            if (enrollTimeline) enrollTimeline.innerHTML = "";
            resetEnrollSlots();
            updateEnrollProgress();
            if (enrollFeedback) {
                enrollFeedback.textContent = "Enrollment reset. Begin repetition 1.";
                enrollFeedback.style.color = "var(--muted)";
            }
        });
    }

    const btnDemoFillEnrollment = document.getElementById("btn-demo-fill-enrollment");
    if (btnDemoFillEnrollment) {
        btnDemoFillEnrollment.addEventListener("click", () => {
            const username = document.getElementById("enroll-username").value.trim();
            if (!username) {
                if (enrollFeedback) {
                    enrollFeedback.textContent = "Please enter a Username before simulating repetitions.";
                    enrollFeedback.style.color = "var(--danger)";
                }
                const userInput = document.getElementById("enroll-username");
                if (userInput) userInput.focus();
                return;
            }

            const baseTemplate = [
                0.092, 0.210, 0.118,
                0.088, 0.165, 0.077,
                0.081, 0.145, 0.064,
                0.086, 0.180, 0.094,
                0.110, 0.240, 0.130,
                0.098, 0.190, 0.092,
                0.084, 0.150, 0.066,
                0.078, 0.160, 0.082,
                0.085, 0.170, 0.085,
                0.090, 0.220, 0.130,
                0.105
            ];

            state.enrollment.samples = [];
            for (let s = 0; s < 19; s++) {
                const sample = baseTemplate.map(val => {
                    const jitter = (Math.random() - 0.5) * 0.025;
                    return Number(Math.max(0.04, val + jitter).toFixed(6));
                });
                state.enrollment.samples.push(sample);
            }

            if (typeof enrollRecorder !== "undefined" && enrollRecorder) {
                enrollRecorder.reset();
            }
            if (enrollInput) {
                enrollInput.value = "";
                enrollInput.disabled = false;
                enrollInput.focus();
            }
            if (enrollSuccessBanner) enrollSuccessBanner.style.display = "none";
            resetEnrollSlots();
            updateEnrollProgress();

            if (enrollFeedback) {
                enrollFeedback.textContent = "19 baseline repetitions pre-filled. Type repetition 20 in the Passphrase Arena to complete model training!";
                enrollFeedback.style.color = "var(--accent)";
            }
        });
    }

    if (btnGotoVerify) {
        btnGotoVerify.addEventListener("click", () => {
            switchTab("tab-verify");
        });
    }

    // -------------------------------------------------------------------------
    // WORKSPACE 2: LIVE VERIFICATION & REAL-TIME THRESHOLD CALIBRATION
    // -------------------------------------------------------------------------
    const verifyUserSelect = document.getElementById("verify-user-select");
    const verifyInput = document.getElementById("verify-typing-input");
    const verifySlots = document.querySelectorAll("#verify-char-slots .char-slot");
    const verifyFeedback = document.getElementById("verify-feedback");
    const verifyCard = document.getElementById("verify-passphrase-card");
    const verifyTimeline = document.getElementById("verify-cadence-timeline");
    const thresholdSlider = document.getElementById("threshold-slider");
    const sliderThresholdVal = document.getElementById("slider-threshold-val");
    const sliderSecurityMode = document.getElementById("slider-security-mode");
    const tradeoffFarVal = document.getElementById("tradeoff-far-val");
    const tradeoffFrrVal = document.getElementById("tradeoff-frr-val");

    // Decision display elements
    const decisionCard = document.getElementById("verification-decision-card");
    const decisionText = document.getElementById("decision-text");
    const decisionSubtext = document.getElementById("decision-subtext");
    const metricScore = document.getElementById("metric-score");
    const metricThreshold = document.getElementById("metric-threshold");
    const metricMargin = document.getElementById("metric-margin");
    const metricConfidence = document.getElementById("metric-confidence");
    const deviationsList = document.getElementById("deviations-list");

    let lastVerifiedFeatures = null;

    function resetVerifySlots() {
        highlightNextKey(verifySlots, 0);
    }

    async function onVerifyUserChanged(username) {
        if (!username) return;
        const badge = document.getElementById("active-user-badge");
        if (badge) badge.textContent = `Active: ${username}`;

        try {
            const tradeoffData = await apiRequest(`/tradeoff/${username}`);
            state.tradeoffs = tradeoffData.tradeoffs || [];
            const baseThresh = tradeoffData.current_threshold;

            const minT = typeof tradeoffData.min_threshold === "number" ? tradeoffData.min_threshold : -0.20;
            const maxT = typeof tradeoffData.max_threshold === "number" ? tradeoffData.max_threshold : 0.20;
            const step = typeof tradeoffData.step === "number" ? tradeoffData.step : 0.005;

            if (thresholdSlider) {
                thresholdSlider.min = minT;
                thresholdSlider.max = maxT;
                thresholdSlider.step = step;
                thresholdSlider.value = baseThresh;
            }

            updateThresholdDisplay(baseThresh);
        } catch (e) {
            console.error("Failed to load tradeoff for user:", e);
        }
    }

    if (verifyUserSelect) {
        verifyUserSelect.addEventListener("change", (e) => {
            onVerifyUserChanged(e.target.value);
        });
    }

    function updateThresholdDisplay(val) {
        const numVal = parseFloat(val);
        if (sliderThresholdVal) sliderThresholdVal.textContent = numVal.toFixed(4);

        if (state.tradeoffs.length > 0) {
            let closest = state.tradeoffs[0];
            let minDiff = Math.abs(closest.threshold - numVal);
            for (const pt of state.tradeoffs) {
                const diff = Math.abs(pt.threshold - numVal);
                if (diff < minDiff) {
                    minDiff = diff;
                    closest = pt;
                }
            }

            if (tradeoffFarVal) tradeoffFarVal.textContent = `${closest.far_percent.toFixed(1)}%`;
            if (tradeoffFrrVal) tradeoffFrrVal.textContent = `${closest.frr_percent.toFixed(1)}%`;

            if (sliderSecurityMode) {
                if (closest.far_percent < 20.0) {
                    sliderSecurityMode.textContent = "Mode: Zero-Trust Security";
                    sliderSecurityMode.style.color = "var(--accent)";
                } else if (closest.frr_percent < 20.0) {
                    sliderSecurityMode.textContent = "Mode: High Convenience";
                    sliderSecurityMode.style.color = "var(--success)";
                } else {
                    sliderSecurityMode.textContent = "Mode: Balanced (EER)";
                    sliderSecurityMode.style.color = "var(--accent)";
                }
            }
        }
    }

    if (thresholdSlider) {
        thresholdSlider.addEventListener("input", (e) => {
            updateThresholdDisplay(e.target.value);
            if (lastVerifiedFeatures && verifyUserSelect && verifyUserSelect.value) {
                reEvaluateAttempt(verifyUserSelect.value, lastVerifiedFeatures, parseFloat(e.target.value));
            }
        });
    }

    async function reEvaluateAttempt(username, features, activeThreshold) {
        try {
            const res = await apiRequest("/verify", "POST", {
                username: username,
                features: features,
                threshold: activeThreshold
            });
            renderVerificationResult(res);
        } catch (err) {
            console.error(err);
        }
    }

    function renderVerificationResult(res) {
        if (!decisionCard) return;
        decisionCard.style.display = "flex";

        if (res.is_accepted) {
            decisionCard.className = "verdict-card accept";
            if (decisionText) decisionText.textContent = "ACCEPT";
            if (decisionSubtext) decisionSubtext.textContent = `Identity Verified: Keystroke cadence matches ${res.username}'s profile.`;
        } else {
            decisionCard.className = "verdict-card reject";
            if (decisionText) decisionText.textContent = "REJECT";
            if (decisionSubtext) decisionSubtext.textContent = `Anomaly Detected: Typing rhythm deviates significantly from ${res.username}'s profile.`;
        }

        if (metricScore) metricScore.textContent = res.score.toFixed(4);
        if (metricThreshold) metricThreshold.textContent = res.threshold.toFixed(4);
        const marginSign = res.distance_margin >= 0 ? "+" : "";
        if (metricMargin) {
            metricMargin.textContent = `${marginSign}${res.distance_margin.toFixed(4)}`;
            metricMargin.style.color = res.is_accepted ? "var(--success)" : "var(--danger)";
        }
        if (metricConfidence) metricConfidence.textContent = `${res.confidence_percent.toFixed(1)}%`;

        if (deviationsList) {
            deviationsList.innerHTML = "";
            if (res.top_deviations && res.top_deviations.length > 0) {
                res.top_deviations.forEach(dev => {
                    const tag = document.createElement("span");
                    tag.className = "deviation-tag";
                    const sign = dev.diff_ms >= 0 ? "+" : "";
                    tag.textContent = `${dev.feature}: ${sign}${dev.diff_ms}ms (${dev.direction})`;
                    deviationsList.appendChild(tag);
                });
            }
        }
    }

    if (verifyInput && verifySlots.length > 0) {
        const verifyRecorder = new KeystrokeRecorder({
            onProgress: (data) => {
                highlightNextKey(verifySlots, data.targetIndex);
            },
            onKeyTiming: (data) => {
                if (verifyTimeline) {
                    const bar = document.createElement("div");
                    bar.className = "cadence-bar";
                    const h = Math.min(48, Math.max(6, data.holdMs / 5));
                    bar.style.height = `${h}px`;
                    bar.title = `${data.key}: ${data.holdMs}ms`;
                    verifyTimeline.appendChild(bar);
                    if (verifyTimeline.children.length > 22) {
                        verifyTimeline.removeChild(verifyTimeline.firstChild);
                    }
                }
            },
            onError: (msg) => {
                if (verifyFeedback) {
                    verifyFeedback.textContent = msg;
                    verifyFeedback.style.color = "var(--danger)";
                }
                verifyInput.value = "";
                resetVerifySlots();
                verifyRecorder.reset();
                setTimeout(() => {
                    if (verifyFeedback) {
                        verifyFeedback.textContent = "Type .tie5Roanl then press Return";
                        verifyFeedback.style.color = "var(--muted)";
                    }
                }, 3000);
            },
            onComplete: async (result) => {
                const username = verifyUserSelect ? verifyUserSelect.value : "";
                if (!username) {
                    if (verifyFeedback) {
                        verifyFeedback.textContent = "Please select a User profile to verify against.";
                        verifyFeedback.style.color = "var(--danger)";
                    }
                    verifyInput.value = "";
                    resetVerifySlots();
                    verifyRecorder.reset();
                    return;
                }

                if (verifyFeedback) {
                    verifyFeedback.textContent = "Analyzing typing dynamics against trained model...";
                    verifyFeedback.style.color = "var(--accent)";
                }
                lastVerifiedFeatures = result.features;

                try {
                    const currentThresh = thresholdSlider ? parseFloat(thresholdSlider.value) : undefined;
                    const res = await apiRequest("/verify", "POST", {
                        username: username,
                        features: result.features,
                        threshold: currentThresh,
                        is_impostor_attempt: false
                    });

                    renderVerificationResult(res);
                    if (verifyFeedback) {
                        verifyFeedback.textContent = `Verification complete for ${username}.`;
                        verifyFeedback.style.color = res.is_accepted ? "var(--success)" : "var(--danger)";
                    }
                } catch (err) {
                    if (verifyFeedback) {
                        verifyFeedback.textContent = `Verification error: ${err.message}`;
                        verifyFeedback.style.color = "var(--danger)";
                    }
                } finally {
                    verifyInput.value = "";
                    resetVerifySlots();
                    verifyRecorder.reset();
                    verifyInput.focus();
                }
            }
        });

        verifyInput.addEventListener("keydown", (e) => verifyRecorder.handleKeyDown(e));
        verifyInput.addEventListener("keyup", (e) => verifyRecorder.handleKeyUp(e));
        verifyInput.addEventListener("focus", () => {
            if (verifyCard) verifyCard.classList.add("focused");
            if (verifyRecorder.downEvents.length === 0) {
                verifyInput.value = "";
            }
        });
        verifyInput.addEventListener("blur", () => verifyCard && verifyCard.classList.remove("focused"));
    }

    // -------------------------------------------------------------------------
    // WORKSPACE 3: ADVERSARIAL IMPERSONATOR ARENA
    // -------------------------------------------------------------------------
    const impVictimSelect = document.getElementById("impersonate-target-user");
    const impFriendName = document.getElementById("impersonate-friend-name");
    const impInput = document.getElementById("impersonate-typing-input");
    const impSlots = document.querySelectorAll("#impersonate-char-slots .char-slot");
    const impFeedback = document.getElementById("impersonate-feedback");
    const impCard = document.getElementById("impersonate-passphrase-card");
    const impTimeline = document.getElementById("impersonate-cadence-timeline");
    const impDecisionCard = document.getElementById("impersonate-decision-card");
    const impDecisionText = document.getElementById("impersonate-decision-text");
    const impDecisionSubtext = document.getElementById("impersonate-decision-subtext");
    const impScore = document.getElementById("impersonate-score");
    const impThreshold = document.getElementById("impersonate-threshold");
    const impOutcome = document.getElementById("impersonate-outcome");
    const impLogTbody = document.getElementById("impersonation-log-tbody");
    const btnSimulateImpostor = document.getElementById("btn-simulate-impostor");

    function resetImpSlots() {
        highlightNextKey(impSlots, 0);
    }

    function addImpersonationLogRow(entry) {
        if (!impLogTbody) return;
        if (impLogTbody.children.length === 1 && impLogTbody.children[0].querySelector(".table-empty")) {
            impLogTbody.innerHTML = "";
        }

        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td><code>${entry.timestamp}</code></td>
            <td><strong>${entry.target}</strong></td>
            <td>${entry.friend}</td>
            <td class="nums"><code>${entry.score}</code></td>
            <td class="nums"><code>${entry.threshold}</code></td>
            <td>
                <span class="tag-rank ${entry.isBlocked ? 'tag-best' : 'tag-runner'}">
                    ${entry.isBlocked ? 'DEFENSE SECURE (BLOCKED)' : 'ALERT (BREACH)'}
                </span>
            </td>
        `;
        impLogTbody.prepend(tr);
    }

    async function submitImpersonatorFeatures(features, friendLabel) {
        const victim = impVictimSelect ? impVictimSelect.value : "";
        const friend = friendLabel || (impFriendName ? impFriendName.value.trim() : "") || "Adversary 01";

        if (!victim) {
            if (impFeedback) {
                impFeedback.textContent = "Please select a Victim profile to attack.";
                impFeedback.style.color = "var(--danger)";
            }
            if (impInput) impInput.value = "";
            resetImpSlots();
            return;
        }

        if (impFeedback) {
            impFeedback.textContent = "Evaluating motor dynamics against victim profile...";
            impFeedback.style.color = "var(--warning)";
        }

        try {
            const res = await apiRequest("/verify", "POST", {
                username: victim,
                features: features,
                is_impostor_attempt: true,
                tester_label: friend
            });

            if (impDecisionCard) impDecisionCard.style.display = "flex";

            if (!res.is_accepted) {
                // BLOCKED
                if (impDecisionCard) impDecisionCard.className = "verdict-card reject";
                if (impDecisionText) impDecisionText.textContent = "BLOCKED";
                if (impDecisionSubtext) impDecisionSubtext.textContent = `Attack Repelled! ${friend}'s motor timing failed verification against ${victim}.`;
                if (impOutcome) {
                    impOutcome.textContent = "REJECTED (SAFE)";
                    impOutcome.style.color = "var(--success)";
                }
            } else {
                // BREACHED
                if (impDecisionCard) impDecisionCard.className = "verdict-card accept";
                if (impDecisionText) impDecisionText.textContent = "BREACHED";
                if (impDecisionSubtext) impDecisionSubtext.textContent = `Warning: ${friend}'s typing rhythm was close enough to pass threshold.`;
                if (impOutcome) {
                    impOutcome.textContent = "ACCEPTED (BREACH)";
                    impOutcome.style.color = "var(--danger)";
                }
            }

            if (impScore) impScore.textContent = res.score.toFixed(4);
            if (impThreshold) impThreshold.textContent = res.threshold.toFixed(4);

            addImpersonationLogRow({
                timestamp: res.timestamp,
                target: victim,
                friend: friend,
                score: res.score.toFixed(4),
                threshold: res.threshold.toFixed(4),
                isBlocked: !res.is_accepted
            });

            if (impFeedback) impFeedback.textContent = `Impersonation trial complete for ${friend}.`;
        } catch (err) {
            if (impFeedback) {
                impFeedback.textContent = `Impersonation test error: ${err.message}`;
                impFeedback.style.color = "var(--danger)";
            }
        } finally {
            if (impInput) impInput.value = "";
            resetImpSlots();
            if (typeof impRecorder !== "undefined" && impRecorder) {
                impRecorder.reset();
            }
        }
    }

    let impRecorder = null;
    if (impInput && impSlots.length > 0) {
        impRecorder = new KeystrokeRecorder({
            onProgress: (data) => {
                highlightNextKey(impSlots, data.targetIndex);
            },
            onKeyTiming: (data) => {
                if (impTimeline) {
                    const bar = document.createElement("div");
                    bar.className = "cadence-bar";
                    const h = Math.min(48, Math.max(6, data.holdMs / 5));
                    bar.style.height = `${h}px`;
                    bar.style.background = "var(--danger)";
                    impTimeline.appendChild(bar);
                    if (impTimeline.children.length > 22) {
                        impTimeline.removeChild(impTimeline.firstChild);
                    }
                }
            },
            onError: (msg) => {
                if (impFeedback) {
                    impFeedback.textContent = `Typo: ${msg}`;
                    impFeedback.style.color = "var(--danger)";
                }
                impInput.value = "";
                resetImpSlots();
                impRecorder.reset();
            },
            onComplete: (result) => {
                submitImpersonatorFeatures(result.features, null);
                impRecorder.reset();
            }
        });

        impInput.addEventListener("keydown", (e) => impRecorder.handleKeyDown(e));
        impInput.addEventListener("keyup", (e) => impRecorder.handleKeyUp(e));
        impInput.addEventListener("focus", () => {
            if (impCard) impCard.classList.add("focused");
            if (impRecorder && impRecorder.downEvents.length === 0) {
                impInput.value = "";
            }
        });
        impInput.addEventListener("blur", () => impCard && impCard.classList.remove("focused"));
    }

    // Button to simulate an impostor attack using synthetic cadence variance
    if (btnSimulateImpostor) {
        btnSimulateImpostor.addEventListener("click", () => {
            const victim = impVictimSelect ? impVictimSelect.value : "";
            if (!victim) {
                if (impFeedback) {
                    impFeedback.textContent = "Please select a Victim profile to attack first.";
                    impFeedback.style.color = "var(--danger)";
                }
                return;
            }

            // Generate synthetic adversary vector (noticeable neuromuscular timing shift)
            const syntheticFeatures = [];
            // 11 Hold times (skewed higher/lower than typical user)
            for (let i = 0; i < 11; i++) {
                syntheticFeatures.push(0.045 + Math.random() * 0.12);
            }
            // 10 Down-Down flight times
            for (let i = 0; i < 10; i++) {
                syntheticFeatures.push(0.12 + Math.random() * 0.35);
            }
            // 10 Up-Down flight times
            for (let i = 0; i < 10; i++) {
                syntheticFeatures.push(0.03 + Math.random() * 0.22);
            }

            const attackerAlias = impFriendName && impFriendName.value.trim() ? impFriendName.value.trim() : "Adversary Bot";
            submitImpersonatorFeatures(syntheticFeatures, attackerAlias);
        });
    }

    // -------------------------------------------------------------------------
    // Initial App Boot
    // -------------------------------------------------------------------------
    purgeAllInputFields();
    resetHeroSlots();
    resetEnrollSlots();
    resetVerifySlots();
    resetImpSlots();
    loadUsers();
});
