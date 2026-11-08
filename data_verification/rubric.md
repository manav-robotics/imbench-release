# Episode Rating Rubric

The 1-5 rubric used to rate the released teleoperated episodes.

# ⭐ Scores

## **5 — Perfect**

* Smooth, stable, efficient
* Clear strategy, no hesitation
* No unnecessary motion or corrections
* Fully controlled contacts

👉 Gold-standard training data

---

## **4 — Good**

* Correct strategy
* Minor inefficiencies or small corrections
* Mostly smooth and stable

👉 High-quality, usable

---

## **3 — Acceptable**

* Task done but:

  * Noticeable inefficiency or instability
  * Object dropped but picked up again 
  * Multiple small corrections
* Execution is somewhat messy

👉 Use sparingly

---

## **2 — Poor**

* Success is fragile
* Many corrections / retries
* Jerky or poorly controlled motion

👉 Low-quality, avoid if possible

---

## **1 — Very Poor**

* Accidental or non-reproducible success
* Erratic, unstable, or chaotic behavior
* Teaches bad behavior

👉 Filter out

---

# ⚖️ What to Judge

* **Motion quality** (smooth vs jerky)
* **Efficiency** (direct vs wasteful)
* **Control** (stable vs near-failure)
* **Strategy clarity** (intentional vs random)

---

# 🚫 Immediate Downgrades

* Near drops / instability
* Repeated failed attempts
* Long hesitation
* Unnecessary collisions

---

# ✅ Quick Heuristic

* **5** → copy this
* **4** → good
* **3** → messy
* **2** → barely worked
* **1** → discard
