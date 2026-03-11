# Presentation Guide: CFO Estimation for OFDM-Based Wireless Communication Systems Under Impulsive Noise

**Conference Presentation Date:** March 13, 2026
**Presenter (you):** Shivansh Rai
**Institution:** ABV-IIITM Gwalior

> Read it once, end to end, before the presentation. Everything you need to know — the story, the concepts, the slide-by-slide script, and likely questions — is here.

---

## Table of Contents

1. [The Big Picture — What Is This Paper About?](#1-the-big-picture)
2. [Background Concepts (Plain English)](#2-background-concepts-plain-english)
3. [The Problem This Paper Solves](#3-the-problem-this-paper-solves)
4. [The Solution — What Was Proposed](#4-the-solution)
5. [The Results — Did It Work?](#5-the-results)
6. [Slide-by-Slide Talking Points](#6-slide-by-slide-talking-points)
7. [Key Numbers to Remember](#7-key-numbers-to-remember)
8. [Glossary of Terms](#8-glossary-of-terms)
9. [Likely Questions and Answers](#9-likely-questions-and-answers)
10. [Presentation Logistics and Tips](#10-presentation-logistics-and-tips)

---

## 1. The Big Picture

**What does this paper do, in one sentence?**

> This paper proposes a smarter way to calibrate wireless receivers in environments with "spiky" interference, using a new deep learning model that is more robust than previous methods.

---

**Imagine this analogy:**

You are trying to tune a radio to exactly the right frequency. If the radio dial is even slightly off, you get static and cross-talk from other stations — this is the real-world equivalent of the problem studied here. Now imagine that problem is made worse because someone is occasionally firing a very loud noise gun nearby (impulsive noise — like a spark or industrial machine). Standard auto-tuning systems get confused and break down under these conditions.

This paper builds a smarter auto-tuning system (using deep learning / AI) that stays accurate even when the noise gun goes off.

---

**Why does this matter?**

- Modern wireless systems (5G, IoT, smart factories, urban networks) use a technology called **OFDM** (explained below). OFDM is very sensitive to even tiny frequency mismatches.
- In real industrial or urban environments, interference is often "spiky" (impulsive), not smooth. Standard algorithms assume smooth noise and fail badly under spiky noise.
- The proposed method — called **CAD-HuberNet** — handles both cases well and is nearly **10 dB better** than the previous best approach under bad noise conditions (10 dB ≈ 10× better performance in engineering terms).

---

## 2. Background Concepts (Plain English)

Read this section carefully. These are the concepts you will need to explain if someone asks a basic question.

---

### 2.1 OFDM — Orthogonal Frequency Division Multiplexing

**What it is:** OFDM is the wireless transmission technology used in 4G, 5G, Wi-Fi, and digital TV. Instead of sending data on one radio frequency, it splits data across hundreds or thousands of smaller sub-frequencies (called **subcarriers**) simultaneously.

**Why it is popular:**

- Handles multi-path interference well (signals bouncing off buildings, walls, etc.)
- Very efficient — packs a lot of data into available bandwidth

**The big weakness:** All those subcarriers must be perfectly aligned (orthogonal) to each other. If the receiver's internal clock is even slightly off-frequency compared to the transmitter, everything falls apart — subcarriers start interfering with each other. This is called **Inter-Carrier Interference (ICI)**.

---

### 2.2 CFO — Carrier Frequency Offset

**What it is:** The small frequency mismatch between the transmitter and receiver.

**Why it happens:**

- Hardware imperfections — the tiny crystal oscillators inside phones and base stations are not perfectly identical
- Doppler shift — when you are moving (e.g., in a car), the frequency of received signals shifts slightly, just like how a siren sounds higher-pitched when approaching and lower-pitched when moving away

**Why it is a problem:** Even a tiny CFO destroys the orthogonality of OFDM subcarriers, causing ICI, which severely degrades communication quality (more errors, slower speeds).

**The goal of this paper:** Accurately *estimate* CFO from the received signal so it can be corrected. This is called **CFO estimation**. The paper proposes a *blind* estimator — meaning it does not rely on known "training" symbols embedded in the data (which would waste bandwidth).

---

### 2.3 Impulsive Noise

**What it is:** Noise that is mostly quiet but occasionally produces very large, sudden spikes.

**Where it comes from:** Electrical motors, switching power supplies, lightning, industrial machines, vehicle ignition systems.

**Why it is a problem for standard methods:** Standard algorithms assume noise follows a Gaussian (bell-curve) distribution — smooth, symmetric, with no extreme outliers. Impulsive noise has a **heavy tail** — extremely large spikes occur far more often than a Gaussian would predict.

**Math model used:** The paper models impulsive noise using the **Symmetric α-Stable (SαS) distribution**. The key parameter is **α (alpha)**:

- α = 2: reduces to ordinary Gaussian noise (benign)
- α < 2: impulsive heavy-tailed noise (more impulsive as α decreases toward 1)
- α = 1.4 was used in simulations (moderately impulsive)
- α = 1.3 was used for the worst case

---

### 2.4 Deep Learning (DL) — the AI Part

**What it is:** Deep learning uses artificial neural networks — software systems loosely inspired by the brain — to learn patterns from data. Once trained on examples, the network can process new inputs and produce accurate outputs.

**How it is used here:** Instead of using mathematical formulas to estimate CFO, the network learns from thousands of simulated (input signal → correct CFO value) examples and then estimates CFO for new received signals.

**Key DL terms you might hear:**

| Term                                         | Simple meaning                                                                                                                                       |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| **CNN** (Convolutional Neural Network) | Specialized at finding patterns in sequential/spectral data                                                                                          |
| **DNN** (Deep Neural Network)          | A network with many stacked layers                                                                                                                   |
| **Attention mechanism**                | A part of the network that learns to focus on the most important parts of the input signal and ignore less useful parts                              |
| **Loss function**                      | The score that tells the network how wrong its answer is — it tries to minimize this during training                                                |
| **MMSE / MSE loss**                    | The most common loss function — penalizes errors proportional to their*square*. Very sensitive to large spikes                                    |
| **Huber loss**                         | A smarter loss function that behaves like MSE for small errors but switches to a gentler linear penalty for large errors (explained in detail below) |
| **Backpropagation**                    | The process by which a neural network adjusts its internal weights to reduce errors                                                                  |
| **Gradient explosion**                 | When impulsive spikes enter squared-error loss, they create enormous gradients that destabilize training — the network breaks                       |

---

### 2.5 Why Standard DL Fails Under Impulsive Noise

When MMSE (squared error) loss is used:

- A single impulsive spike that is, say, 10× larger than normal creates a loss 100× larger (because it is squared).
- This generates a huge gradient in backpropagation.
- The network parameters get wildly updated — training becomes unstable (oscillates, diverges).
- The resulting trained model performs poorly.

---

### 2.6 The Huber Loss — the Key Innovation

The Huber loss is a hybrid between MSE and MAE (Mean Absolute Error):

```
For small errors (|e| ≤ δ):  Loss = ½ × e²    ← same as MSE, precise
For large errors (|e| > δ):  Loss = δ|e| - ½δ²  ← linear, like MAE
```

**What this means in practice:**

- Small, normal estimation errors are penalized precisely with a squared term (good accuracy)
- Large outlier errors (from impulsive noise) are penalized gently — linearly — instead of squaring
- The gradient for outliers is **capped at δ**, so it never explodes
- Training becomes stable even when impulsive noise is severe

Think of it as: "For small mistakes, we are strict. For catastrophically large mistakes caused by noisy data, we forgive somewhat and do not overreact."

---

## 3. The Problem This Paper Solves

**Existing work's gap:** Previous deep learning methods for CFO estimation (e.g., ResNet-based, CNN-based) are trained using MMSE loss, which assumes Gaussian noise. Under impulsive (SαS) noise:

- Training oscillates and is unstable
- The final model has poor CFO estimation accuracy
- No prior work combined the CAD architecture with a robust (Huber) loss function

**This paper's contribution:**

1. Take the best existing DL architecture for CFO estimation — the **CAD (CNN-Attention-DNN)** model
2. Replace its MMSE loss with **Huber loss**
3. Show through simulation that this simple but powerful change yields dramatically better performance under impulsive noise, while matching MMSE performance under Gaussian noise

---

## 4. The Solution

### 4.1 System Setup (What is Simulated)

- OFDM system with **N = 128 subcarriers**
- Cyclic prefix length **Ncp = 16**
- Modulation: **64-QAM** (a way of encoding data into radio signals)
- Channel: **frequency-selective with L = 3 taps** (signal bounces 3 times)
- CFO range: uniformly random between **-0.5 and +0.5** (normalized)
- Noise: **SαS with α = 1.4, γ = 0.1** (moderately impulsive)
- Dataset: **20,000 OFDM symbols**, 70% training / 30% validation
- Training: Adam optimizer, batch size 1000, up to 2000 epochs

---

### 4.2 The CAD-HuberNet Architecture (What the Model Looks Like)

The model has **two parallel processing streams** that work together:

#### Stream 1 (Upper) — Squared Spectrum

- Takes the received signal r[n]
- Squares it: |r[n]|²
- Applies Fourier Transform (FFT): converts from time domain to frequency domain
- Takes the magnitude: |FFT(|r[n]|²)|
- This amplifies the cyclic frequency patterns caused by CFO

#### Stream 2 (Lower) — Magnitude + Phase

- Takes the received signal r[n]
- Applies Fourier Transform
- Takes both magnitude AND phase angle of the result
- Preserves phase information that helps resolve ambiguities in frequency estimation

**Why two streams?** Each stream captures different aspects of the signal. Together, they give the network a richer view than either alone.

#### Inside each stream, the data flows through:

1. **Convolutional layers (CNN)** — extract local spectral patterns
2. **Triple Residual Stacks (TRS)** — deep layers with skip connections that prevent vanishing gradient problems (a common deep learning problem where deep networks stop learning)
3. **Attention mechanism** — the network learns to focus on the reliable, clean subcarriers and ignore the ones corrupted by impulsive spikes
4. **Reshape** — restructures the data for the attention layer

#### After both streams:

5. **Concatenation** — outputs from both streams are joined together
6. **Fully Connected DNN** — four layers (1024 → 560 → 512 → 16 neurons) that map the combined features to a single number
7. **Output: ε̂** — the estimated CFO value

The entire network is trained with **Huber loss** instead of MMSE loss.

---

### 4.3 The Cyclic Prefix (CP) — Why It Helps Blind Estimation

OFDM signals include a "cyclic prefix" — a copy of the end of each OFDM symbol is pasted onto the front. This is normally used to handle multi-path channel effects. For blind CFO estimation, the correlation between the CP and its copy carries implicit frequency offset information the network can learn from.

---

## 5. The Results

Three experiments were run:

### Experiment 1: Training Convergence (MSE vs. Epoch)

**What it shows:** How quickly and smoothly each model learns during training.

**Result:**

- **CAD-MMSE:** Oscillating, unstable loss curve — the squared error magnifies impulsive spikes and causes gradient explosion
- **CAD-HuberNet:** Smooth, monotonically decreasing loss curve — stable training even with impulsive noise

**What to say:** "The Huber loss makes training stable and reliable. The MMSE model oscillates because every impulsive spike it sees creates a huge, destabilizing gradient."

---

### Experiment 2: Steady-State MSE vs. SNR

**What it shows:** Final estimation accuracy across different signal-to-noise ratios (SNR). Higher SNR = less noise overall, so performance should improve for both models.

**Key result (α = 1.4, moderately impulsive):**

| SNR   | CAD-HuberNet MSE     | CAD-MMSE MSE | Gain    |
| ----- | -------------------- | ------------ | ------- |
| 15 dB | **−19.21 dB** | −9.42 dB    | ~10 dB  |
| 25 dB | **−25.32 dB** | −17.69 dB   | ~7.6 dB |

Lower MSE (more negative) = better estimation accuracy.

**What to say:** "At 15 dB SNR, our model achieves −19.21 dB MSE versus −9.42 dB for the baseline — a nearly 10 dB improvement, which is roughly a 10× reduction in mean squared estimation error."

---

### Experiment 3: MSE vs. α (Impulsiveness Level) at SNR = 20 dB

**What it shows:** How each model handles different levels of impulsiveness.

**Key results:**

| α value | Noise Type        | CAD-HuberNet MSE     | CAD-MMSE MSE  |
| -------- | ----------------- | -------------------- | ------------- |
| 1.3      | Very impulsive    | **−1.38 dB**  | ≈ 0 dB       |
| 2.0      | Gaussian (benign) | **−20.50 dB** | ≈ −20.50 dB |

**Critical insight:** As α → 2.0 (Gaussian), both models converge to the same performance. This proves that using Huber loss costs you nothing under normal (Gaussian) conditions — it only helps under impulsive conditions.

**What to say:** "In highly impulsive conditions (α = 1.3), our model is significantly better. When noise is Gaussian (α = 2.0), our model matches the baseline. There is no trade-off — you get robustness for free."

---

## 6. Slide-by-Slide Talking Points

The presentation has **14 slides**. Here is exactly what to say for each one.

---

### Slide 1: Title Slide

**Say:**

> "Good [morning/afternoon]. This presentation is about estimating carrier frequency offset — a key synchronization parameter — in OFDM wireless systems that operate under impulsive noise. The work is by Shivansh Rai, Sandesh Jain, Mahesh Chaudhari, and Poornima Singh Thakur from ABV-IIITM Gwalior and TIET Patiala."

---

### Slide 2: Outline

**Say:**

> "I'll walk you through the motivation and problem, the system and noise model, the proposed deep learning architecture, how we train it robustly, and finally the simulation results."

---

### Slide 3: Motivation and Problem Statement

**Two blocks on this slide:**

**Block 1 — "The Impact of CFO":**

> "OFDM is the backbone of 5G and Wi-Fi. To work correctly, all of its sub-frequencies must be perfectly aligned. Carrier frequency offset — caused by hardware imperfections or motion — destroys this alignment and causes inter-carrier interference, which degrades communication quality."

**Block 2 — "Limitations of Existing DL Estimators":**

> "Deep learning has been applied to estimate CFO, but all prior work uses MMSE loss — which penalizes errors by squaring them. Under impulsive noise, the squared loss massively amplifies the large error spikes, causing gradient explosion and unstable training. This is the core problem we solve."

---

### Slide 4: System Model Overview (Figure)

This slide shows the full signal flow diagram. Point to blocks as you describe them:

> "Here is the system. On the left, message bits are modulated using QAM and then transformed by IFFT — this creates the OFDM signal. The signal is upconverted to a carrier frequency and transmitted wirelessly. The channel introduces multi-path fading. At the receiver, the signal picks up a frequency offset — CFO — and is also corrupted by impulsive noise. Our proposed model, shown in orange on the right, receives the corrupted signal r[n] and outputs the CFO estimate ε̂."

---

### Slide 5: OFDM Signal Model

**Block 1 — Transmitted Signal:**

> "The transmitted signal is a sum of N modulated subcarriers, created using an Inverse FFT operation, with a cyclic prefix appended."

**Block 2 — Received Signal:**

> "After propagating through the channel and picking up CFO ε and noise w[n], the received signal r[n] has a phase rotation term e to the power j·2π·ε·n/N multiplied in. The job of our estimator is to figure out ε from r[n] alone, without any known pilot symbols — that's what makes it a blind estimator."

---

### Slide 6: Impulsive Noise Model

> "We model the noise using the Symmetric alpha-Stable (SαS) distribution, which is the standard mathematical model for impulsive interference in industrial and urban environments."

> "The characteristic function has a parameter α (alpha) that controls impulsiveness. When α equals 2, we get ordinary Gaussian noise. When α is less than 2, the distribution has heavy tails — meaning rare but extremely large spikes occur. Crucially, for α less than 2, the variance is mathematically infinite. This means any estimator that relies on squared errors — like MMSE — will diverge, because those huge spikes contribute infinitely to the squared loss."

---

### Slide 7: Multi-Modal Input Feature Representation

> "Instead of feeding the raw received signal directly to the neural network, we construct two processed representations."

> "The upper stream takes the magnitude of the FFT of the squared received signal. Squaring is a nonlinear operation that amplifies the cyclic frequency components created by the CFO, making them stand out more clearly."

> "The lower stream takes both the magnitude and phase of the FFT of the raw signal. This preserves phase information, which helps the network distinguish between different possible CFO values that might otherwise look similar."

> "Together, these two streams give the network complementary spectral information — like viewing a scene with two different types of cameras."

---

### Slide 8: Proposed CAD-HuberNet Architecture (Figure)

Point to the diagram as you describe it:

> "The architecture is a dual-stream design. Both the upper and lower streams enter their respective branches simultaneously. In each branch, convolutional layers extract spectral patterns, followed by Triple Residual Stacks — which are deep layers with skip connections that prevent the vanishing gradient problem — followed by a reshape and an attention layer. The attention layer assigns higher weights to reliable, clean subcarriers and lower weights to those corrupted by impulsive spikes. The outputs of both branches are then concatenated and passed through four fully connected layers that produce the final scalar CFO estimate ε̂."

---

### Slide 9: CAD-HuberNet Key Components

> "Three key components work together:"

> "First, Triple Residual Stacks. The skip connection adds the input directly to the output of each layer. This means even if a layer learns nothing useful, the signal still passes through unchanged. This prevents deep networks from getting worse as they get deeper."

> "Second, the Attention Mechanism. It computes a score for each time step in the feature sequence, then normalizes these scores into weights that sum to one. The weighted sum becomes a single context vector representing the most important information from the entire sequence. In the context of impulsive noise, corrupted features receive low attention weights and clean features receive high weights."

> "Third, the DNN Regression Head. The context vectors from both streams are concatenated and passed through four dense layers that ultimately output a single number — the estimated CFO."

---

### Slide 10: Robust Optimization via Huber Loss

> "The most important innovation is replacing the standard MMSE loss with Huber loss."

> "The Huber penalty function is quadratic — like MSE — for small errors below a threshold δ. But for large errors above δ, it becomes linear. This means that massive impulsive spikes, which would have contributed squared-error values to the loss under MMSE, now only contribute a linear amount."

> "The gradient of the Huber loss is also bounded. For large errors, the gradient is capped at ±δ, no matter how large the error actually is. This is called gradient saturation. It prevents impulsive noise from blowing up the gradient during backpropagation, ensuring stable convergence."

> "The tunable threshold δ controls where the transition happens. This single parameter is all that is different from standard training."

---

### Slide 11: Results — Convergence and SNR Performance

*Two plots side by side on this slide.*

**Left plot (MSE vs Epoch):**

> "During training, CAD-MMSE oscillates severely — impulsive spikes cause the loss to spike up repeatedly. CAD-HuberNet shows a smooth, stable, monotone decrease. This confirms that Huber loss prevents gradient explosion during training."

**Right plot (MSE vs SNR):**

> "In steady state, our model consistently outperforms the baseline. At 15 dB SNR, we achieve −19.21 dB MSE versus −9.42 dB — that's almost a 10 dB gain, meaning roughly 10 times lower mean squared error. The gap is present across the entire SNR range."

---

### Slide 12: Results — Robustness to Impulsiveness (α)

> "Here we fix SNR at 20 dB and vary α — the impulsiveness of the noise."

> "In the highly impulsive regime at α = 1.3, our model achieves MSE around −1.38 dB while the MMSE baseline is near 0 dB. This shows our model handles severe impulsiveness gracefully."

> "Importantly, as α approaches 2.0, the Gaussian case, both models converge to nearly identical performance. This confirms that using Huber loss does not hurt you when the environment is benign — it only helps when noise is impulsive. This is a key result: there is no performance penalty for using the robust loss function."

---

### Slide 13: Conclusion

> "To summarize: we proposed CAD-HuberNet, a blind CFO estimator for OFDM systems under impulsive noise."

> "The key contributions are: first, we use a dual-stream CNN-Attention-DNN architecture that captures both power-spectral and phase information. Second, we replace the standard MMSE training with Huber loss, which bounds gradients and prevents training instability. Third, our simulations show approximately 10 dB performance gain over the state-of-the-art DL baseline under highly impulsive conditions, while matching its performance under Gaussian noise."

> "This makes the proposed method directly applicable to 5G, industrial IoT, and other next-generation wireless systems that must operate in environments with impulsive interference."

---

### Slide 14: Thank You / Questions

> "Thank you for your attention. I'd be happy to take questions."

*(Stay calm. If you don't know an answer, say: "That is a great question — I'll note it for the authors to address.")*

---

## 7. Key Numbers to Remember

Write these on your hand if needed. These are the facts most likely to be asked about.

| Fact                               | Value                                     |
| ---------------------------------- | ----------------------------------------- |
| Number of subcarriers              | N = 128                                   |
| Cyclic prefix length               | Ncp = 16                                  |
| Modulation scheme                  | 64-QAM                                    |
| Channel taps                       | L = 3                                     |
| CFO range                          | Uniform [−0.5, 0.5]                      |
| Noise model                        | SαS, α = 1.4, γ = 0.1                  |
| Training dataset size              | 20,000 OFDM symbols                       |
| Training / Validation split        | 70% / 30%                                 |
| Batch size                         | 1,000                                     |
| Max epochs                         | 2,000                                     |
| Optimizer                          | Adam, initial LR = 10⁻³                 |
| DNN layer sizes                    | 1024 → 560 → 512 → 16                  |
| Gain vs baseline at 15 dB SNR      | ~10 dB (−19.21 dB vs −9.42 dB)          |
| Gain vs baseline at 25 dB SNR      | ~7.6 dB (−25.32 dB vs −17.69 dB)        |
| Gain at α = 1.3                   | CAD-HuberNet: −1.38 dB vs MMSE: ≈0 dB   |
| Performance at α = 2.0 (Gaussian) | Both ≈ −20.50 dB — Huber costs nothing |

---

## 8. Glossary of Terms

| Term                                            | Plain-English meaning                                                                                                                    |
| ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **OFDM**                                  | Wireless technology splitting data across hundreds of sub-frequencies simultaneously (used in 5G, Wi-Fi)                                 |
| **CFO**                                   | The frequency mismatch between transmitter and receiver — like a radio dial slightly off station                                        |
| **ICI**                                   | Inter-Carrier Interference — the static caused when CFO destroys OFDM's frequency alignment                                             |
| **Blind estimation**                      | Estimating CFO without sending any known "calibration" test signals (saves bandwidth)                                                    |
| **Impulsive noise**                       | Noise that is mostly quiet but has rare, large spikes (from motors, lightning, etc.)                                                     |
| **SαS distribution**                     | Mathematical model for impulsive noise; controlled by parameter α                                                                       |
| **α (alpha)**                            | Controls impulsiveness: α = 2 is Gaussian, smaller α = more impulsive spikes                                                           |
| **AWGN**                                  | Additive White Gaussian Noise — standard smooth bell-curve noise (unrealistic for many real environments)                               |
| **DL / Deep Learning**                    | Teaching a neural network to solve a problem by showing it many examples                                                                 |
| **CNN**                                   | Convolutional Neural Network — good at finding patterns in sequences                                                                    |
| **DNN**                                   | Deep Neural Network — many stacked layers that learn complex mappings                                                                   |
| **Attention mechanism**                   | Network component that learns which parts of the input matter most                                                                       |
| **Residual connection / Skip connection** | A shortcut in the network that adds input directly to output, preventing vanishing gradients                                             |
| **Loss function**                         | Score measuring how wrong the model's prediction is; the model tries to minimize it                                                      |
| **MMSE loss**                             | Squared error loss — standard but sensitive to impulsive outliers                                                                       |
| **Huber loss**                            | Hybrid loss: quadratic for small errors, linear for large ones — robust to outliers                                                     |
| **Gradient explosion**                    | When impulsive spikes cause enormous gradients that destabilize neural network training                                                  |
| **Adam optimizer**                        | A standard algorithm for adjusting network weights during training                                                                       |
| **MSE**                                   | Mean Squared Error — the metric used to evaluate estimation accuracy; lower (more negative in dB) is better                             |
| **SNR**                                   | Signal-to-Noise Ratio — how much stronger the useful signal is than the noise; higher SNR = better conditions                           |
| **dB**                                    | Decibel — a logarithmic scale. −19 dB MSE is much better than −9 dB MSE. A difference of 10 dB ≈ 10× in linear scale                |
| **CP / Cyclic Prefix**                    | A guard interval in OFDM; a copy of the end of a symbol pasted at the beginning                                                          |
| **QAM**                                   | Quadrature Amplitude Modulation — a way to encode more bits per radio signal                                                            |
| **IFFT / FFT**                            | Inverse/Fast Fourier Transform — mathematical operations to convert between time-domain and frequency-domain representations of signals |
| **TRS**                                   | Triple Residual Stacks — three stacked residual (skip-connection) convolutional blocks                                                  |
| **CAD**                                   | CNN-Attention-DNN — the architecture name for the proposed model                                                                        |
| **δ (delta)**                            | The Huber loss threshold — the boundary between quadratic and linear penalty regions                                                    |

---

## 9. Likely Questions and Answers

### Q1: Why do you use a blind estimator instead of a pilot-based one?

**A:** Pilot-based estimators embed known test signals into every data transmission. While accurate, this wastes bandwidth — you send fewer actual data bits per second. A blind estimator uses only the statistical properties of the signal itself (like the cyclic prefix redundancy) to estimate CFO, wasting no bandwidth. In high-throughput applications like 5G, this spectral efficiency matters.

---

### Q2: Why was the Huber loss chosen over other robust loss functions?

**A:** The Huber loss is well-studied, computationally simple, and provides a smooth, differentiable function everywhere (which is important for gradient-based training). The threshold δ provides a single, intuitive tuning parameter. Other robust criteria like the Maximum Correntropy Criterion or Versoria criterion exist and are mentioned in the paper, but the Huber loss was chosen for its balance of robustness and simplicity. Future work could explore those alternatives.

---

### Q3: How was δ (the Huber threshold) chosen?

**A:** The paper does not report a specific methodology for selecting δ beyond citing the Adam optimizer during training. In practice, δ is typically tuned as a hyperparameter using validation set performance. You can say: "δ was treated as a hyperparameter optimized during the training process."

---

### Q4: How does your model compare to classical (non-DL) CFO estimators?

**A:** Classical methods like the van de Beek ML estimator or covariance-matrix-based methods also appear in the related work. They rely on statistical assumptions (like Gaussian noise or large numbers of OFDM symbols) that fail under impulsive noise. DL-based methods learn directly from data and can adapt. This paper focuses on comparing within the DL category (CAD-HuberNet vs. CAD-MMSE) as the primary comparison, noting that DL already outperforms classical methods in prior work.

---

### Q5: Can this model work in real-time?

**A:** Once trained, the neural network's inference is extremely fast — just a forward pass through the network for each received OFDM symbol. The training is done offline. So yes, post-training deployment is suitable for real-time operation. Hardware implementation was not part of this work, which focused on algorithm design and simulation.

---

### Q6: Why 128 subcarriers and 64-QAM? Is this realistic?

**A:** These are standard simulation parameters. 128 subcarriers is a practical size used in research, and 64-QAM is used in real 5G systems. The results would be expected to generalize to larger systems (256, 512, 1024 subcarriers) as the paper demonstrates a fundamental robustness principle, not a system-specific trick.

---

### Q7: What does 20,000 training samples mean — is that enough?

**A:** 20,000 OFDM symbols is a reasonable dataset size for this task. Each symbol is a synthetic simulation under known channel and noise conditions. The authors observed that the model converged well and showed good generalization on the 30% validation set with this amount of data. In practice, synthetic OFDM data is cheap to generate.

---

### Q8: What is the computational complexity compared to CAD-MMSE?

**A:** The architecture is identical to CAD-MMSE — only the loss function used during training changes. Inference complexity is therefore exactly the same. Training is also similarly complex. The only difference is gradient clipping behavior during backpropagation.

---

### Q9: Would this work under time-varying (non-quasi-stationary) CFO?

**A:** This paper assumes quasi-stationary CFO — meaning it is roughly constant over the observation window. Time-varying CFO would be a more challenging scenario and is noted as a potential direction for future work.

---

### Q10: What are the future research directions?

**A:** Natural extensions include:

- Testing on real hardware with real impulsive noise measurements
- Extending to time-varying CFO scenarios
- Combining CFO estimation with joint channel estimation
- Applying similar robust loss function ideas to other wireless parameter estimation problems (e.g., timing offset, channel estimation)


### Key Message to Leave the Audience With

> **A simple change — replacing the squared-error loss with Huber loss — makes a deep learning CFO estimator robust to impulsive noise with no cost under normal conditions, achieving a ~10 dB performance gain in impulsive environments.**

---

*Good luck with the presentation! The work is solid, the results are clear, and the story is straightforward.*
