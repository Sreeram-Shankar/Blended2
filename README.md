# Blended2 — Stiffness + Curvature-Controlled Hybrid Implicit ODE Solver

Blended2 is an experimental adaptive ODE solver that combines **method blending** and **curvature-based step size control** into a single, unified framework.

Unlike traditional adaptive solvers, Blended2 does **not** rely on:
- Embedded Runge–Kutta pairs
- Method-specific local truncation error estimators
- Discrete stiffness detection or method switching

Instead, it introduces **two orthogonal adaptivity mechanisms**:

1. **Method blending** driven by a stiffness proxy  
2. **Step size control** driven by geometric curvature

This separation allows Blended2 to adapt *how* it advances the solution independently from *how far* it advances in time.

---

## Core Idea

Blended2 advances the solution by solving a **single nonlinear equation per step** that smoothly interpolates between two implicit methods:

- **Implicit Midpoint (Gauss–Legendre 1-stage)**  
  - Energy-preserving  
  - Symplectic  
  - Ideal for nonstiff or moderately stiff dynamics  

- **BDF2 (Second-order Backward Differentiation Formula)**  
  - Strongly damping  
  - L-stable  
  - Ideal for stiff dynamics  

Rather than switching between these methods, Blended2 blends them continuously using a **stiffness-dependent weight**.

---

Good catch — GitHub README rendering is the classic trap here.

GitHub does not render LaTeX math unless you use images or very limited extensions, so the right fix is to rewrite the math in readable plain-text / pseudo-math form, while keeping it precise.

Below is a fully fixed, GitHub-native README for Blended2, with:

No LaTeX

Clear ASCII math

Preserved rigor

Clean GitHub formatting

Copy-paste ready

# Blended2 — Curvature-Controlled Hybrid Implicit ODE Solver

Blended2 is an experimental adaptive ODE solver that combines **method blending** and **curvature-based step size control** into a single unified framework.

Unlike traditional adaptive solvers, Blended2 does **not** rely on:
- Embedded Runge–Kutta pairs
- Method-specific local truncation error (LTE) estimators
- Explicit stiffness detection or method switching

Instead, it introduces **two orthogonal adaptivity mechanisms**:

1. **Method blending** driven by a stiffness proxy  
2. **Step size control** driven by geometric curvature  

This separation allows Blended2 to adapt *how* it advances the solution independently from *how far* it advances in time.

---

## Core Idea

Blended2 advances the solution by solving **one nonlinear implicit equation per step** that smoothly interpolates between two implicit methods:

- **Implicit Midpoint (Gauss–Legendre 1-stage)**
  - Symplectic
  - Energy-preserving
  - Ideal for nonstiff or weakly stiff problems

- **BDF2 (Second-order Backward Differentiation Formula)**
  - Strongly damping
  - L-stable
  - Ideal for stiff dynamics

Rather than switching between these methods, Blended2 **blends them continuously** using a stiffness-dependent weight.

---

## Blended Residual Formulation

Each step solves a single residual equation of the form:



R(y_{n+1}) =
(1 - a) * R_GL1(y_{n+1})

a * R_BDF2(y_{n+1}) = 0


Where:
- `R_GL1` is the implicit midpoint residual
- `R_BDF2` is the BDF2 residual
- `a` is a blending parameter in the range [0, 1]

This produces **one implicit solve per step**, not two competing methods.

---

## Stiffness Proxy and Blending Weight

The blend parameter `a` is computed from a **dimensionless stiffness proxy**:



sigma = h * || f_n - f_{n-1} || / ( || y_n - y_{n-1} || + eps )


This quantity estimates how rapidly the vector field is changing relative to the state over the current step.

The blending weight is defined as:



a = sigma^p / (1 + sigma^p)


Properties:
- `sigma << 1`  →  `a ≈ 0`   (pure implicit midpoint)
- `sigma >> 1`  →  `a ≈ 1`   (pure BDF2)
- Smooth transition for intermediate regimes

No hard thresholds. No discrete switching.

---

## Orthogonal Step Size Control: Curvature, Not LTE

Blended2 uses **curvature-based step size adaptivity**, not truncation-error control.

Instead of asking:
> “Is my formal method order satisfied?”

Blended2 asks:
> “Is the solution trajectory bending too sharply for this step size?”

---

### Curvature Defect

The controller estimates a second-derivative-like quantity along the trajectory.

Conceptually:



y_ddot ≈ ( f_{n+1} - f_n ) / h


The curvature defect scales like:



E ≈ 0.5 * h^2 * || y_ddot ||


This measures **geometric curvature**, not algebraic local truncation error.
---

### Why Curvature?

- Independent of method order
- Sensitive to sharp transients
- Natural for stiff and nonstiff problems alike
- Decouples adaptivity from numerical order

This makes curvature control compatible with **method blending**, where no single formal order exists.

---

## Step Acceptance and Rejection

A step is rejected if:
- The curvature defect exceeds tolerance
- The nonlinear (Newton) solve fails to converge

On rejection:
- Step size is reduced
- The same method blend is retried
- Hard failure only occurs after repeated rejection

---

## Full Algorithm Outline

1. **Initialization**
   - First step taken with pure implicit midpoint
2. **For each step**
   - Estimate stiffness proxy \( \sigma \)
   - Compute blending weight \( a(\sigma) \)
   - Form blended nonlinear residual
   - Solve via Newton iteration
   - Estimate curvature defect
3. **Accept or reject**
   - Update solution
   - Adapt step size via PI controller on curvature

Method blending and step size control remain **fully independent**.

---

## What Blended2 Is

- A research solver
- A hybrid implicit integrator
- A testbed for nontraditional adaptivity
- A demonstration that LTE-based control is not mandatory

---

## What Blended2 Is Not

- A production-grade solver
- A replacement for CVODE or Radau5
- A black-box integrator
- A formally optimized scheme

This solver prioritizes **conceptual clarity and experimentation**.

---

## Example Usage

```python
from blended2_adaptive import solve_blended2_adaptive
import numpy as np

def f(t, y):
    return -1000*y + np.sin(t)

t, y, info = solve_blended2_adaptive(
    f,
    t_span=(0.0, 2.0),
    y0=[1.0],
    h0=1e-2
)
```
---
**Returned diagnostics may include:**

- Blending weights

- Stiffness proxy history

- Curvature norms

- Newton iteration counts

- Rejection statistics

## Expected Behavior
- Smooth dynamics → behaves like implicit midpoint

- Moderate stiffness → blended behavior

- Strong stiffness → converges toward BDF2

- Large curvature → aggressive step reduction

- Flat regions → rapid step growth

This solver often behaves very differently from classical adaptive RK solvers — by design.

## Files
- blended2_fixed.py — fixed-step blended solver

- blended2_adaptive.py — curvature-adaptive solver
