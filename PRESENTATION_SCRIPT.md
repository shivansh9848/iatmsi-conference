---
## Slide 1 — Title Slide

*(The slide shows the paper title, author names, and affiliations. Stand at the podium, wait for the audience to settle, then begin.)*

**Say:**

> "Good morning / Good afternoon, everyone. My name is [YOUR NAME], and I am presenting this work on behalf of the authors — Shivansh Rai, Sandesh Jain, Mahesh Chaudhari, and Poornima Singh Thakur from ABV-IIITM Gwalior and TIET Patiala.
>
> The title of this paper is: *CFO Estimation for OFDM-Based Wireless Communication Systems Under Impulsive Noise.*
>
> In simple terms — this work builds a smarter deep learning model to fix a fundamental wireless synchronization problem that gets worse in noisy industrial environments. I'll walk you through the motivation, the system, the proposed solution, and the results."

*(Pause for 2 seconds, then advance.)*
---
## Slide 2 — Outline

*(The slide shows the section headings as a table of contents: Introduction → System & Noise Model → Proposed Architecture → Robust Optimization → Simulation Results → Conclusion.)*

**Say:**

> "Here is the roadmap for today's talk. I'll start with the motivation — why CFO matters and why existing deep learning approaches fall short. Then I'll describe the system and noise model we work with. After that, I'll walk through the proposed CAD-HuberNet architecture, explain the key training innovation — the Huber loss — and finally present simulation results that validate the approach. Let's dive in."

*(Advance.)*

---

## Slide 3 — Motivation and Problem Statement

*(Two blocks on screen: top block in blue titled "The Impact of CFO", bottom alert block in red titled "Limitations of Existing DL Estimators".)*

**Say — Top Block:**

> "Every wireless device you use today — your phone on 4G LTE or 5G NR, your laptop on Wi-Fi — uses a transmission technology called OFDM, Orthogonal Frequency Division Multiplexing. OFDM is popular because it splits data across hundreds of sub-frequencies simultaneously, making it spectral-efficient and robust against signal reflections off buildings and walls.
>
> But OFDM has one critical weakness: all those sub-frequencies must be perfectly aligned. If the receiver's internal clock is even slightly off-frequency compared to the transmitter — we call this a Carrier Frequency Offset, or CFO — the alignment breaks. Sub-frequencies start leaking into each other, causing what we call Inter-Carrier Interference, ICI. This directly degrades the bit error rate, meaning more of your data gets corrupted. CFO arises all the time — from tiny hardware mismatches in oscillators, and from Doppler shifts when a device is moving."

*(Point to the red alert block.)*

**Say — Bottom Block:**

> "Researchers have turned to deep learning to estimate and correct this CFO. These DL models work well under standard noise assumptions, but here is the problem — they are all trained using MMSE loss, which is a squared-error objective.
>
> Under impulsive noise — which is common in industrial IoT, factories, and urban environments with motors, power lines, and switching equipment — the noise produces occasional very large spikes. When you square a large spike in your loss function, you get an enormous number. This sends a huge, destabilizing gradient back through the network during training — what we call gradient explosion. The model oscillates, fails to converge, and performs poorly.
>
> This is the exact limitation this paper addresses."

*(Advance.)*

---

## Slide 4 — System Model Overview

*(A figure fills the slide showing: Message Bits → QAM Modulation → IFFT+CP → Upconversion → Antenna → Channel → Antenna → CFO phase rotation → Impulsive Noise addition → Proposed CAD-Huber Model → CFO estimate ε̂.)*

**Say:**

> "Here is the full system we consider. Let me walk you through it from left to right.
>
> On the transmitter side: message bits are first modulated using 64-QAM — a standard scheme that encodes multiple bits per symbol. The modulated symbols are then converted from frequency domain to time domain using an Inverse FFT, and a cyclic prefix is appended. The signal is upconverted to a carrier frequency and transmitted over the air.
>
> In the middle: the signal passes through a frequency-selective fading channel — meaning the signal bounces off multiple surfaces and arrives at the receiver via multiple paths.
>
> At the receiver: two things corrupt the signal. First, a CFO phase rotation — the frequency mismatch — wraps around every sample. Second, impulsive noise w[n] is added. This is the received signal r[n] that our model must work with.
>
> On the right, shown in orange: our proposed CAD-Huber model takes this corrupted r[n] and outputs ε̂ — the estimated CFO. Once estimated, this offset can be corrected and the system can function properly. Crucially, this is a *blind* estimator — we do not send any known calibration symbols to help the estimation. That would waste bandwidth."

*(Advance.)*

---

## Slide 5 — Impulsive Noise Model

*(An alert block shows the SαS characteristic function. Below it, two columns: Parameters on the left, Key Implication on the right.)*

**Say:**

> "Now let me describe how we model the noise. In most textbook scenarios, noise is assumed to be Gaussian — smooth, symmetric, with no extreme outliers. But in real industrial and urban environments, noise is *impulsive* — mostly quiet, with sudden large spikes from motors, lightning, or power switching.
>
> We model this using the Symmetric alpha-Stable distribution, or S-alpha-S. Instead of having a closed-form PDF, this distribution is defined by its characteristic function — shown here. The key parameter is alpha.
>
> When alpha equals 2, the distribution reduces to ordinary Gaussian noise — the familiar bell curve. As alpha drops below 2 toward 1, the distribution's tails get heavier and heavier — extreme values occur far more often than any Gaussian would predict.
>
> Here is the critical mathematical implication shown on the right: for any alpha less than 2, the variance of this distribution is *infinite*. Infinite variance means there is no finite average spread of the noise. Any estimator that relies on squared differences — like MMSE — will try to minimize something that is effectively unbounded. The loss explodes, and training fails.
>
> This infinite variance property is why we need a fundamentally different approach — not just a better architecture, but a better loss function."

*(Advance.)*

---

## Slide 6 — CAD-HuberNet Architecture (Figure)

*(A detailed block diagram shows the dual-stream architecture: r[n] enters Feature Extraction, splits into upper (Conv → TRS → Reshape → Attention) and lower (Conv → TRS → Reshape → Attention) branches, both merge at Concat, then pass through FC layers to output ε̂.)*

**Say:**

> "Before I explain the components individually, let me give you the big picture of the architecture.
>
> The received signal r[n] enters a feature extraction block that produces two different representations — an upper stream and a lower stream. Think of it as looking at the signal through two different lenses simultaneously.
>
> Each stream then goes through its own dedicated processing pipeline: convolutional layers that extract spectral patterns, followed by residual stacks, followed by an attention layer that decides which parts of the signal to trust.
>
> The outputs of both streams are then concatenated — joined together — and passed through fully connected layers that ultimately output a single number: ε̂, the estimated CFO.
>
> The entire network — every layer in both streams — is trained using the Huber loss function, which I'll explain shortly. Let's now look at each component."

*(Advance.)*

---

## Slide 7 — Multi-Modal Input Feature Representation

*(Two columns showing the upper stream formula X^(u) and lower stream formula X^(l).)*

**Say:**

> "Why two streams? Because raw time-domain signals are not very informative for frequency estimation — you need to transform them into a more useful representation first.
>
> The upper stream takes the received signal, squares it sample by sample, then applies a Fourier Transform, and takes the magnitude. Why squaring? Squaring is a non-linear operation that amplifies the cyclic frequency patterns created by CFO — it makes the frequency offset 'stand out' more clearly against the noise background. The result is a vector of size S-by-1, where S is the OFDM symbol length.
>
> The lower stream takes the Fourier Transform of the raw received signal and extracts both its magnitude *and* its phase angle. Phase information is critical — without it, you cannot distinguish between different CFO values that might look similar in magnitude alone. The result is a matrix of size S-by-2.
>
> Together, these two streams give the network complementary views: one that amplifies CFO-related power spectral features, and one that preserves precise phase information. Both are needed for accurate estimation."

*(Advance.)*

---

## Slide 8 — CAD-HuberNet: Key Components

*(Three blocks: TRS with skip-connection equation, Attention with softmax equations, DNN Regression Head description.)*

**Say — Block 1, TRS:**

> "The first processing stage inside each stream is the Triple Residual Stack, or TRS. This is a stack of three convolutional blocks, each with a skip connection — meaning the input to a block is added directly to its output. The equation on the slide shows this: Z_l equals the activated convolution plus Z_{l-1}, the original input.
>
> Why does this matter? In deep networks, gradients shrink as they travel backward through many layers during training — a problem called vanishing gradients. The skip connection provides a direct highway for gradients to flow back to early layers unchanged, keeping the entire network trainable regardless of depth."

**Say — Block 2, Attention:**

> "After the TRS, each stream passes through an attention mechanism. This computes a score e_i for each position in the feature sequence — essentially asking 'how useful is this feature?' — using a small learned neural network with a tanh activation. The scores are then normalized using softmax into attention weights alpha_i that sum to one.
>
> The weighted sum of features — called the context vector — summarizes the entire sequence with emphasis on the most informative parts. In the context of impulsive noise, subcarriers corrupted by spikes receive very low attention weights. Clean subcarriers receive high weights. The model effectively learns to ignore the corrupted evidence and focus on what it can trust."

**Say — Block 3, DNN Head:**

> "Finally, the context vectors from both streams are concatenated and fed into four fully connected layers — with 1024, 560, 512, and 16 neurons — that map this rich combined representation to a single scalar output: the estimated normalized CFO ε̂."

*(Advance.)*

---

## Slide 9 — Robust Optimization via Huber Loss

*(Top block shows the Huber penalty function ρ_δ(e) piecewise formula. Bottom alert block shows the gradient ψ(e) piecewise formula.)*

**Say — Top Block:**

> "Now the most important innovation: the loss function. Instead of MMSE — squared error — we use the Huber loss.
>
> The Huber loss is a piecewise function. For small errors below a threshold delta, it behaves exactly like squared error — quadratic, precise, sensitive to small differences. For large errors above delta, it switches to a linear penalty — it grows much more slowly.
>
> Concretely: if MMSE sees an error of 100, it contributes 10,000 to the loss. Huber sees the same error of 100 and contributes only 100 times delta — a fraction of that. Large impulsive spikes are no longer astronomically amplified."

**Say — Bottom Alert Block:**

> "But the real benefit shows up in the gradient — what flows backward during training. The gradient of the Huber loss is bounded. For small errors, the gradient equals the error itself — same as MMSE. For large errors, the gradient is *capped* at plus or minus delta, no matter how large the actual error is. This is called gradient saturation.
>
> This is critical. A single impulsive noise spike that might have sent a gradient of 10,000 into the network under MMSE now sends a gradient of delta — say, 1. The network weights receive a small, controlled update instead of being catastrophically scrambled.
>
> This is why the training curve is stable for our model and oscillating for the MMSE baseline — as you will now see."

*(Advance.)*

---

## Slide 10 — Simulation Setup

*(Two columns: OFDM System Parameters on the left, Training Configuration on the right. Baseline description block at the bottom.)*

**Say:**

> "Before the results, let me briefly state the simulation parameters so you can evaluate the results in context.
>
> On the left: our OFDM system has 128 subcarriers, a cyclic prefix of length 16, 64-QAM modulation, and a frequency-selective channel with 3 taps. The normalized CFO is drawn uniformly at random from minus 0.5 to plus 0.5 for each symbol — covering a wide range of realistic offsets.
>
> On the right: the noise is S-alpha-S with alpha equal to 1.4 and dispersion 0.1 — this represents moderately heavy-tailed impulsive noise. We generated 20,000 synthetic OFDM symbols, split 70/30 into training and validation sets. Training used the Adam optimizer with an initial learning rate of 10 to the minus 3, batch size of 1000, and ran for up to 2000 epochs with early stopping.
>
> The baseline throughout is the existing CAD-MMSE model — identical architecture, trained with MMSE loss instead of Huber loss. The performance metric is MSE in decibels: lower — more negative — is better."

*(Advance.)*

---

## Slide 11 — Results: Convergence and SNR Performance

*(Two plots side by side: left is MSE vs. Epoch training curve, right is Steady-State MSE vs. SNR.)*

**Say — Left Plot (MSE vs. Epoch):**

> "The left plot shows how each model's training loss evolves over epochs. The CAD-MMSE curve — shown in the baseline color — is highly oscillatory. This is gradient explosion in action: every impulsive noise spike in the training data sends a massive gradient through the network, undoing previous learning and causing the model to swing wildly. It eventually converges but it is slow, unstable, and reaches a worse final value.
>
> The CAD-HuberNet curve is smooth and monotonically decreasing. No oscillations. The Huber gradient cap means spikes are simply absorbed without disturbing training. The model converges faster and to a better solution."

**Say — Right Plot (MSE vs. SNR):**

> "The right plot shows final estimation accuracy across a range of SNRs from 0 to 30 dB. Lower MSE — more negative on the y-axis — means more accurate CFO estimation.
>
> CAD-HuberNet outperforms CAD-MMSE across the *entire* SNR range. At 15 dB SNR — a typical operating condition — our model achieves minus 19.21 dB MSE compared to minus 9.42 dB for the baseline. That is a gain of nearly 10 dB, which translates to roughly a 10-times reduction in mean squared estimation error. At 25 dB SNR, we achieve minus 25.32 dB versus minus 17.69 dB — nearly an 8 dB gain. The advantage is consistent and substantial."

*(Advance.)*

---

## Slide 12 — Results: Robustness to Impulsiveness (α)

*(A single plot: steady-state MSE vs. alpha at SNR = 20 dB. Two bullet points below.)*

**Say:**

> "The final experiment asks: how does performance change as we vary the *level* of impulsiveness? We fix SNR at 20 dB and sweep alpha from 1.3 — very impulsive — to 2.0, which is the Gaussian limit.
>
> As alpha decreases, the noise becomes more impulsive and the estimation task becomes harder. Look at the two curves. The CAD-MMSE baseline degrades rapidly as alpha drops — it was never designed for this regime and struggles badly. At alpha equals 1.3, it reaches nearly 0 dB MSE, meaning it is barely better than random guessing.
>
> CAD-HuberNet degrades much more gracefully. At alpha equals 1.3, it achieves minus 1.38 dB MSE — still genuinely estimating the CFO despite very heavy-tailed noise.
>
> Now look at the right side of the plot — alpha approaching 2.0, Gaussian noise. Both curves converge to nearly identical performance, around minus 20 dB. This is the key takeaway: using Huber loss costs you *nothing* in benign conditions. You get the robustness for free. There is no trade-off."

*(Advance.)*

---

## Slide 13 — Conclusion

*(A block titled "Summary of Contributions" with four bullet points.)*

**Say:**

> "To wrap up. This paper made four interconnected contributions.
>
> First, we proposed a robust blind CFO estimation framework — blind meaning no pilot symbols are wasted — specifically designed for heavy-tailed impulsive noise environments that standard methods fail in.
>
> Second, we built a dual-stream CNN-Attention-DNN architecture that extracts both power-spectral and phase features simultaneously from the received signal, giving the network richer input than any single-stream approach.
>
> Third, we replaced MMSE with Huber loss — a targeted, principled fix that bounds gradients during backpropagation and prevents impulsive noise spikes from destabilizing training.
>
> Fourth, and most importantly — our simulations confirm that CAD-HuberNet achieves nearly 10 dB better MSE than the state-of-the-art DL baseline under impulsive conditions, while matching its performance under Gaussian noise. This makes it directly suitable for deployment in 5G, industrial IoT, and urban wireless systems where impulsive interference is a real-world concern.
>
> The core message is simple: one targeted change — swapping the loss function — combined with the right architecture turns a fragile estimator into a robust one."

*(Advance.)*

---

---

## Handling Tough Questions

| If asked...                                         | Say...                                                                                                                                                                                                                             |
| --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| "How was delta chosen?"                             | "Delta is treated as a hyperparameter and tuned on the validation set during training. It controls the transition between the quadratic and linear regimes of the loss."                                                           |
| "Why not use correntropy or other robust criteria?" | "Those are valid alternatives and are cited in the paper. Huber loss was chosen for its simplicity, differentiability everywhere, and single tunable parameter. Exploring other robust criteria is a direction for future work."   |
| "Does this work in real hardware?"                  | "This work presents simulation results. The algorithm complexity is identical to CAD-MMSE at inference time — only the training procedure changes — so hardware deployment is feasible. That validation is a natural next step." |
| "What about time-varying CFO?"                      | "The current model assumes quasi-stationary CFO over the observation window. Extending to time-varying scenarios is an open and important future direction."                                                                       |
| Something you cannot answer                         | "That is a great question. I will make a note of it and ensure the lead author follows up with you directly." — Then write it down visibly.                                                                                       |

---

*End of script. Total estimated delivery time: 13–15 minutes.*
