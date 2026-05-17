# Data Improvement Methods for VLA / Robotics Policy Learning

> **Context**: AIC qualification stage is a **simulation-only** competition (Gazebo). Unlimited rollouts, full simulator state access during training, no real hardware, no human labeling bottleneck, no wear-and-tear cost per episode. All advice below is scoped to this context.

---

## Quick Reference: Methods Ranked by AIC Practicality

| Rank | Method                              | Loss / Objective                   | Key constraint                                                                                   | AIC practicality |
| :--: | ----------------------------------- | :--------------------------------- | ------------------------------------------------------------------------------------------------ | :--------------: |
|  1   | **Domain Randomization**            | None (data augmentation)           | None — just randomize sim params                                                                 |      ★★★★★       |
|  2   | **Sim DAgger** (automated oracle)   | MSE on on-policy states            | Need to build oracle; oracle's own actions may be suboptimal                                     |      ★★★★★       |
|  3   | **Offline RL on frozen VLA tokens** | IQL / CQL on token representation  | Needs reward-labeled rollouts; limited to data distribution (no online exploration)              |      ★★★★★       |
|  4   | **RL Token** (RLT, 2026)            | PPO on frozen VLA + MLP actor      | Needs RL infra, but VLA frozen → no gradient conflict                                            |       ★★★★       |
|  5   | **CR-DAgger**                       | MSE on residual corrections        | More forgiving of imperfect oracle than full DAgger                                              |       ★★★★       |
|  6   | **RECAP** (pi0.6)                   | BC on advantage-filtered timesteps | Can't discover new strategies, only reinforce existing good actions. Needs critic                |       ★★★        |
|  7   | **Noisy Student / Self-training**   | Pseudo-label MSE                   | No reward signal → strictly weaker than offline RL. Reinforces bad behavior if policy is weak    |       ★★★        |
|  8   | **Model-based / MPC**               | Learned dynamics + planning        | High complexity; learned model quality is bottleneck                                             |       ★★★        |
|  9   | **ConRFT** (RSS 2025)               | Unified consistency (BC+Q)         | Incompatible with flow-matching (pi0) — needs consistency training                               |        ★★        |
|  10  | **Sparse RL (PPO)**                 | Policy gradient                    | Incompatible with flow-matching — needs log-probs                                                |        ★★        |
|  —   | **CIFT** (data fidelity)            | None (curation)                    | Passive monitor only — doesn't improve policy. Useful to know about but not a method you execute |        —         |

## 1. Domain Randomization

Randomize sim parameters per training episode. Forces policy to learn invariant representations.

**What to randomize** (all free in Gazebo):
- Task board pose (x, y, yaw)
- NIC rail assignment (which rail, translation, yaw)
- SC rail position (translation)
- Grasp offset (~2mm, ~0.04 rad as per competition spec)
- Lighting intensity and color
- Cable stiffness/damping/friction (physics params)
- Camera noise

**ROI**: Highest for lowest effort. Zero implementation complexity beyond modifying config. Works with any VLA. No assumptions about data quality.

---

## 2. Sim DAgger — with an automated simulator oracle

`L = MSE(π_θ(s), π_oracle(s))` where s ~ states visited by current policy.

**In AIC you have**: full Gazebo state access (plug pose, port location, joint angles, contact forces). You can build a scripted oracle using privileged ground-truth information.

**What the oracle does**: given ground-truth plug pose and target port pose, compute a trajectory (e.g., move plug to port entrance → align → insert). The oracle labels the policy's on-policy states with its own action.

**The key insight**: the oracle doesn't need to be perfect. DAgger's value comes from labeling **on-policy states** (states the policy actually visits, which may be far from the demo distribution), not from action optimality. A mediocre oracle on the right states beats a perfect expert on the wrong states.

**Caveat**: the oracle's own actions could be suboptimal (e.g., a simple motion planner may handle cable dynamics poorly). But the distribution-matching benefit of on-policy labeling often outweighs oracle imperfections.

| Variant | When to use |
|---------|-------------|
| **Full DAgger** (oracle labels every timestep) | Oracle is decent → best |
| **CR-DAgger** (oracle corrects residuals of policy) | Oracle is weak → more forgiving |
| **Diff-DAgger** (label only uncertain frames) | Unnecessary in sim — labeling is free |

---

## 3. Offline RL on Frozen VLA Tokens

**Why this beats Noisy Student**: Both use the same data (rollouts from current policy). But Noisy Student only imitates actions (MSE), while offline RL uses **reward** (AIC score) to learn which actions to prefer.

**What you do**:
1. Generate 1000s of rollouts with domain randomization
2. Label each timestep with its AIC score (or final trajectory score)
3. Freeze the VLA encoder → extract compact token representation per timestep
4. Train an offline RL algorithm (IQL or CQL) on the token dataset: `(state_token, action, reward, next_state_token)`
5. Deploy the token-space policy alongside the frozen VLA

**What offline RL can do that self-imitation cannot**: **Trajectory stitching**. If trajectory A has a great approach but fails at insertion, and trajectory B has a bad approach but succeeds, Noisy Student can't combine them — it only copies actions. Offline RL can learn "use A's approach, then B's insertion" because the value function propagates reward across trajectory boundaries.

**Limitation**: No online exploration — the policy is limited to actions present in the training data distribution. For real exploration, you need online RL (RL Token).

**AIC relevance**: You already have AIC scores for every rollout. The only extra work is training IQL/CQL on the token dataset instead of BC.

Reference: [IQL (2021)](https://arxiv.org/abs/2110.06169), [CQL (2020)](https://arxiv.org/abs/2006.04779)

---

## 4. RL Token (RLT) — Physical Intelligence, March 2026

**How it works**: Distills VLA representations into compact tokens → feeds a tiny MLP actor-critic → runs PPO on the MLP only. The VLA weights **never change**.

**Why this matters for AIC**:
- **No gradient conflict**: VLA provides perception + reference action; separate MLP handles RL. No BC term to fight.
- **No catastrophic forgetting**: VLA frozen throughout RL. Not relevant for AIC (single-task evaluation), but removes a failure mode.
- **Flow-matching compatible**: VLA is frozen, so its architecture doesn't constrain RL method choice.
- **Directly optimizes AIC score**: reward = AIC scoring function (75pt insertion + 12pt speed + 6pt smoothness + ...)

**Cost**: 15 minutes of real-world training in PI's experiments → comparable or less in sim.

**Tradeoff**: needs RL infrastructure (reward function, environment wrapper, PPO implementation). But the sim environment already exists.

Reference: [RL Token (2026)](https://arxiv.org/abs/2604.23073)

---

## 5. CR-DAgger — Residual Corrections for Contact-Rich Tasks

Instead of replacing the policy's action entirely, the oracle provides a **residual correction** on top of the policy's action:

`a_final = a_policy + a_residual`

This is easier for the oracle (it only needs to correct mistakes, not generate full actions from scratch) and more forgiving of oracle imperfections.

**Relevance for AIC**: cable insertion is contact-rich. A scripted oracle that computes "push 2mm left" as a residual is simpler than one that plans the full insertion trajectory.

---

## 6. RECAP — retrain on your own good timesteps

**Used by Physical Intelligence in pi0.6** to reach 92% autonomy across real-world tasks.

**3 steps**:
1. Rollout current policy many times → get (state, action) sequences + AIC scores
2. Train a small value function (critic) from rollout data. Compute per-timestep **advantage** = actual final score - predicted value.
3. Keep only timesteps where advantage > 0. Train on those using BC (MSE).

`Loss = MSE(π_θ(s), a)` only for (s, a) with advantage > 0.

**Why per-timestep**: a failed rollout may have good subsequences (got near port). A successful rollout may have bad subsequences (wobbly approach).

**The ceiling — why it's ranked #7**: RECAP can only reinforce actions the policy *already takes* that happen to lead to good outcomes. If the policy has never succeeded, every rollout is failure → filtering gives nothing. It cannot discover qualitatively new strategies.

**AIC relevance**: critic training needs state information → available from simulator. But the fundamental ceiling limits its value vs. methods that actively explore (DAgger, RL).

---

## 7. Noisy Student / Self-Training

Run current policy on diverse randomized configs → use policy's own predictions as pseudo-labels → retrain.

**Why it's ranked #8**: Same data as offline RL, but no reward signal. Self-imitation can only reinforce what the policy already does — it can't stitch trajectories or learn which actions are better.

**Only useful if**: you have no reward labels. In AIC you do (AIC score per rollout), so offline RL (#3) is strictly better with the same data and similar infra cost.

```
1. Generate 1000 rollouts with randomized configs
2. Keep only top 200 by AIC score (basic filtering)
3. Use policy's actions from those 200 as pseudo-labels
4. Retrain on (state, pseudo_action) pairs using MSE
```

---

## 8. Model-Based / MPC

Learn a dynamics model of the cable+robot system from rollouts, then use Model Predictive Control to plan insertion trajectories.

**Why for cable insertion**: cable dynamics are constrained (bending modes are finite) → a learned model can generalize. MPC can plan precise insertion motions without needing a policy at all.

**Tradeoff**: building an accurate dynamics model for deformable cables is hard. Model bias (the model is wrong in ways the policy exploits) is a well-known failure mode.

---

## 9. Action Representation — Method Selection Depends on It

| VLA Architecture | Compatible methods |
|-----------------|-------------------|
| **Flow-matching** (pi0, pi0.6) | Domain Randomization, Sim DAgger, Offline RL on frozen tokens, RL Token, CR-DAgger, RECAP, Noisy Student, Model-based/MPC |
| Flow-matching | **Incompatible**: PPO, ConRFT, any method needing log π(a\|s) |
| **Diffusion Policy** | Diff-DAgger, RECAP, RL Token, CR-DAgger |
| **ACT** (action-chunked transformer) | Ensemble uncertainty, RECAP, RL Token, PPO (if Gaussian head) |

**Your VLA (pi0-based XVLA) = flow-matching** → PPO, ConRFT, advantage-weighted methods are off the table. RL Token is the best RL option because it freezes the VLA and runs PPO on a separate MLP.

---

## 10. AIC Scoring Reference

| Component | Weight | Condition |
|-----------|:------:|-----------|
| Tier 1: Model validity | 1 pt | Always if model works |
| **Tier 3: Full insertion** | **75 pts** | Correct port |
| Tier 3: Partial insertion | 38-50 pts | Inside port, depth-proportional |
| Tier 3: Proximity | 0-25 pts | Near port, distance-proportional |
| Tier 2: Duration | 0-12 pts | ≤5s=12pt, ≥60s=0pt |
| Tier 2: Smoothness | 0-6 pts | Inverse jerk |
| Tier 2: Efficiency | 0-6 pts | Inverse path length |
| Force penalty | 0 to -12 | >20N for >1s |
| Contact penalty | 0 to -24 | Robot touches enclosure/board |

**Critical threshold**: Tier 2 points (24 pts total) **only awarded if Tier 3 score > 0** (plug near/inside port). Optimization priority:
1. Get near port (unlock Tier 2)
2. Insert fully (75 pts)
3. Optimize speed/smoothness/efficiency

---

## 11. Pipeline for AIC

```
Phase 1: Domain randomization (free, highest ROI)
         randomize board pose, cable params, lighting, grasp offset
         + monitor validation loss for decoherence (CIFT)
         → v1 policy
    ↓
Phase 2: Offline RL on frozen VLA tokens
         generate 10,000 rollouts with domain randomization →
         extract VLA tokens → train IQL/CQL on (token, action, reward) →
         deploy token-space policy alongside frozen VLA
    ↓
Phase 3: Sim DAgger with automated oracle
         build scripted oracle using ground-truth sim state →
         label on-policy states → add to training set →
         retrain → repeat
         (or CR-DAgger if oracle is weak)
    ↓
Phase 4 (optional, if plateaued):
         RL Token (frozen VLA + MLP actor-critic, online PPO)
         reward = AIC scoring function
         directly optimizes the competition metric
    ↓
Phase 5: Force-aware fine-tuning
         shape reward / CR-DAgger to minimize -12 force penalty
```

**Key constraints**:
- VLA = flow-matching (pi0) → PPO, ConRFT unavailable. RL Token, Offline RL on tokens, Sim DAgger work.
- Dataset must include both SFP and SC demonstrations → verify coverage.
- Pre-grasped start reduces distribution shift → BC is more viable than general case.

---

**Note — CIFT (data fidelity)**: CIFT (Stanford, 2025) showed a "decoherence point" where adding more diverse data can hurt performance. Track validation loss on a held-out set; if it increases when adding data, the new data is harming. Not an improvement method itself, but good to be aware of. Reference: [CIFT (2025)](https://arxiv.org/abs/2509.24797)
