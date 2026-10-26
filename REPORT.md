# Keystroke Dynamics Biometric Authentication — Research & Evaluation Report

**Project**: Keyro  
**Author**: Machine Learning Engineering  
**Dataset**: CMU Keystroke Dynamics Benchmark (`DSL-StrongPasswordData.csv`)  
**Target Passphrase**: `.tie5Roanl` (11 keys including Return)  
**Date**: October 2026  

---

## 1. Problem Statement

Traditional knowledge-based authentication mechanisms (alphanumeric passwords and PINs) suffer from severe vulnerabilities:
- **Credential Theft & Phishing**: Passwords can be intercepted, keylogged, shoulder-surfed, or exfiltrated through data breaches.
- **Credential Reuse & Brute Force**: Humans select predictable passwords or reuse identical credentials across platforms.
- **Binary Trust Model**: Once an attacker possesses the plaintext password, the system grants unconditional access.

**Keystroke Dynamics** is a behavioral biometric modality that measures a user's subconscious, neuromuscular typing habits. Because typing involves complex muscle memory, finger flight kinematics, and dwell latencies, two individuals typing the identical password exhibit distinct temporal signatures. 

Keystroke dynamics enables **two-factor security within a single factor**: the user provides *something they know* (the password) and simultaneously verifies *who they are* (their typing cadence), without requiring additional biometric hardware like fingerprint scanners or facial recognition cameras.

---

## 2. Dataset Analysis

The benchmark evaluation uses the canonical **CMU Keystroke Dynamics Benchmark Dataset** collected by Kevin S. Killourhy and Roy A. Maxion (Carnegie Mellon University, 2009).

- **Subjects**: 51 unique human typists.
- **Passphrase**: `.tie5Roanl` followed by Enter (`Return`).
- **Data Volume**: 400 repetitions per subject across 8 distinct sessions (50 repetitions per session spaced over several weeks), totaling **20,400 rows**.
- **Temporal Features (31 dimensions)**:
  - **11 Hold Times ($H.k$):** The dwell duration each key is pressed: $t_{\text{up}}(k) - t_{\text{down}}(k)$.
  - **10 Down-Down Flight Times ($DD.k_1.k_2$):** Dwell-to-dwell latency: $t_{\text{down}}(k_2) - t_{\text{down}}(k_1)$.
  - **10 Up-Down Flight Times ($UD.k_1.k_2$):** Inter-key release-to-press latency: $t_{\text{down}}(k_2) - t_{\text{up}}(k_1)$.
  - *Mathematical relationship:* $DD.k_1.k_2 = H.k_1 + UD.k_1.k_2$.

All measurements in the dataset are recorded with millisecond-precision floating-point seconds (e.g., $0.1491\text{ s} = 149.1\text{ ms}$).

---

## 3. Methodology & Anomaly Detection Models

Because authentication is inherently an **open-set anomaly detection** problem—a production server cannot possess training samples from every future attacker—we evaluate one-class classifiers trained exclusively on the genuine user's typing data.

### 3.1 Evaluated Algorithms

1. **Scaled Manhattan Distance Baseline**:
   For training feature vectors $X \in \mathbb{R}^{N \times D}$, compute the mean vector $\mu \in \mathbb{R}^D$ and standard deviation vector $\sigma \in \mathbb{R}^D$. For a test sample $x$, the anomaly score is the mean standardized $L_1$ distance:
   $$\text{Score}_{\text{Manhattan}}(x) = \frac{1}{D} \sum_{j=1}^D \frac{|x_j - \mu_j|}{\sigma_j + \epsilon}$$
   Higher scores represent greater deviations from the user's habitual template.

2. **One-Class Support Vector Machine (OC-SVM)**:
   Uses an RBF kernel $K(x, x') = \exp(-\gamma \|x - x'\|^2)$ to construct an optimal enclosing hypersphere in reproducing kernel Hilbert space (RKHS) around the genuine user's standardized feature distribution. With outlier fraction parameter $\nu = 0.05$ and `gamma='scale'`:
   $$\text{Score}_{\text{OC-SVM}}(x) = -\text{decision\_function}(x)$$

3. **Isolation Forest**:
   An ensemble of 100 randomized isolation trees. Anomalous typing attempts exhibit shorter average path lengths $h(x)$ from root to leaf:
   $$\text{Score}_{\text{IForest}}(x) = -\text{score\_samples}(x)$$

### 3.2 Evaluation Protocol (Killourhy & Maxion, 2009)

For each of the 51 subjects:
- **Training Set**: The first 200 repetitions of the genuine user ($rep \le 200$).
- **Genuine Test Set**: The remaining 200 repetitions of that user ($rep > 200$).
- **Impostor Test Set**: The first 5 test repetitions from each of the other 50 subjects ($50 \times 5 = 250$ zero-effort impostor attempts per user).
- **Metrics**: 
  - **FAR (False Accept Rate)**: Proportion of impostor attempts accepted ($\text{score} < \theta$).
  - **FRR (False Reject Rate)**: Proportion of genuine attempts rejected ($\text{score} \ge \theta$).
  - **EER (Equal Error Rate)**: The threshold $\theta^*$ where $\text{FAR}(\theta^*) = \text{FRR}(\theta^*)$.
  - **AUC-ROC**: Area Under the Receiver Operating Characteristic curve.

---

## 4. Benchmark Results & Comparative Analysis

The offline benchmark was executed across all 51 subjects. Results are summarized below:

| Anomaly Detector | Mean EER (%) | Std EER (%) | Median EER (%) | Mean AUC | Mean FAR (%) | Mean FRR (%) | Ranking |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **One-Class SVM (RBF)** | **14.12%** | **± 9.54%** | **11.35%** | **0.9113** | **14.12%** | **14.12%** | 🥇 **Best Overall** |
| **Scaled Manhattan** | 14.34% | ± 10.21% | 11.80% | 0.9118 | 14.33% | 14.34% | 🥈 Competitive Baseline |
| **Isolation Forest** | 15.15% | ± 9.97% | 12.85% | 0.9097 | 15.15% | 15.15% | 🥉 3rd Place |

### 4.1 Key Findings

1. **One-Class SVM achieved the lowest EER (14.12%)** and the lowest standard deviation (±9.54%), indicating greater robustness across subjects with varied typing tempos.
2. **Scaled Manhattan Baseline is remarkably competitive (14.34% EER)**:
   Despite its conceptual simplicity, the feature-standardized Manhattan distance performs almost on par with kernel SVMs. This occurs because keystroke timing distributions are approximately unimodal with feature-specific variances (e.g. hold times have much smaller variance than flight transitions involving the Shift key).
3. **High AUC Across All Models (~0.91)**:
   All three algorithms exhibit strong discriminative power, proving that zero-effort impostors who know the password can still be reliably flagged.
4. **Inter-Subject Variance**:
   Certain typists (e.g., subject `s005`) achieved an outstanding EER of **6.0%** (AUC = 0.9844) due to highly rhythmic, repetitive muscle memory. In contrast, subjects with erratic typing habits (e.g., subject `s007`) exhibited an EER of **30.65%**, illustrating the inherent biometric variability between individuals.

---

## 5. Live Application Architecture & UX

To make this technology accessible, we integrated the ML pipeline into a high-performance web application:
- **Client Side (`keystroke.js`)**: Captures millisecond timestamps using the browser's high-precision `performance.now()` clock. Validates key ordering (`.tie5Roanl` + Enter) and computes identical 31 features on the fly.
- **Enrollment Phase**: Requires 20 repetitions (a practical compromise between user fatigue and statistical stability). Fits the top-performing One-Class SVM model and stores the user profile.
- **Dynamic Decision Slider**: Allows the user to move a threshold slider in real time to visualize the security-convenience trade-off:
  - *Strict / Banking Mode* ($\theta \downarrow$): Minimizes FAR to < 5%, preventing impostors at the expense of requiring genuine users to type with near-perfect cadence.
  - *Convenient Unlock* ($\theta \uparrow$): Minimizes FRR to < 5%, accommodating minor typing fluctuations while still filtering crude impersonations.
- **Friend Impersonator Challenge**: Provides an intuitive security demonstration where a friend enters the correct password; the system flags their temporal deviations and logs the attack in an audit scoreboard.

---

## 6. Practical Limitations & Edge Cases

When deploying keystroke dynamics in production environments, several fundamental constraints must be considered:

### 6.1 Small Enrollment Sample Size (20 vs. 200 Repetitions)
- While the CMU benchmark utilizes 200 training samples per user, demanding 200 repetitions in a commercial application causes severe user fatigue.
- With only 20 enrollment samples, estimating feature variance ($\sigma_j$) is subject to sample noise. Regularization or Bayesian shrinkage toward global keystroke priors is recommended for small sample enrollments.

### 6.2 Hardware & Keyboard Latency Variance
- **Key Switch Mechanics**: Membrane laptop keyboards (low travel, ~1.5mm) produce significantly shorter dwell times ($H.k$) than mechanical keyboards (high travel, 4.0mm, tactile switches).
- **Polling Rate Jitter**: Standard office USB keyboards poll at 125 Hz (8ms jitter), while gaming keyboards poll at 1000 Hz (1ms jitter). Bluetooth and wireless keyboards introduce variable packet latency.
- *Remedy:* The model must store per-device profiles or normalize relative flight times.

### 6.3 Intra-User Physiological & Environmental Drift
- **Circadian Rhythm & Fatigue**: A user typing early in the morning or late at night exhibits longer dwell and flight times.
- **Cognitive Load & Stress**: Typing under time pressure alters cadence.
- **Physical Factors**: Cold fingers, hand fatigue, or minor hand injuries substantially alter Up-Down flight times.
- *Remedy:* Implement adaptive template updating (online learning) where verified genuine attempts gently update the user's baseline running average.

### 6.4 Synthetic & Scripted Playback Attacks
- A sophisticated adversary who records a user's keydown/keyup events via client-side malware can replay identical microsecond timestamps via automated input injection (`pyautogui` or virtual keyboards).
- *Remedy:* Keystroke dynamics should serve as an unobtrusive secondary biometric (2FA or continuous authentication) rather than a standalone primary authentication factor.

---

## 7. Conclusion & Recommendations

The Keyro project demonstrates that keystroke dynamics is an effective, zero-hardware behavioral biometric capable of filtering password-sharing impostors with an **Equal Error Rate of ~14.1%** and an **AUC of 0.91**.

For production deployments:
1. **Use One-Class SVM with RBF kernel** as the primary classifier.
2. **Enforce an initial enrollment of 20 repetitions**, supplemented with adaptive profile updates during successful verifications.
3. **Allow threshold tuning** based on the sensitivity of the protected resource (e.g., lower threshold for wire transfers, higher threshold for casual session unlocks).
4. **Deploy as step-up authentication**: When typing score falls in the ambiguous zone near the threshold, prompt the user for an out-of-band factor (e.g., TOTP or biometric WebAuthn).
