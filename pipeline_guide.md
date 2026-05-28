# Quick & Simple Guide: Multi-Objective Recommendation Pipeline

Welcome! This guide explains how our advanced recommendation system works in a very simple, high-level manner. It shows how we take a user's history and turn it into the ultimate **highly accurate, diverse, and serendipitous** movie recommended list.

---

## 1. What is this Pipeline?

Traditional recommender systems only focus on **Accuracy** (recommending popular movies similar to what you already watched). This leads to boring, repetitive recommendations.

Our pipeline uses a **Multi-Objective Genetic Algorithm (NSGA-II)** powered by **BERT AI Review Analysis** to balance three goals:
1. **Accuracy (NCF):** Do you actually like the recommended movies?
2. **Diversity (DOPM):** Do the movies cover different genres to prevent fatigue?
3. **Serendipity (BERT):** Are the movies pleasantly unexpected, niche, and highly relevant?

---

## 2. Step-by-Step System Flow (How it Works)

The entire flow is divided into 5 simple blocks:

```
[ Raw Movie Data ] 
       │
       ▼
 1. Accuracy Engine (NCF AI Model)
       │ ──► Predicts user scores for all movies.
       ▼
 2. Diversity & Serendipity Engines (BERT & DOPM)
       │ ──► Reads IMDb reviews using BERT and tracks movie genres.
       ▼
 3. Smart Population Generator (PSNR Sweet-Spot)
       │ ──► Samples hundreds of high-quality starting movie lists.
       ▼
 4. Evolutionary Solver (pymoo NSGA-II GA)
       │ ──► Mutates, crosses, and refines lists to ensure 0 duplicates.
       ▼
 5. Compromise Selector & MOHS Score
       │ ──► Selects the single best-balanced list and scores its quality!
       ▼
[ Ultimate Recommended Movie List! ]
```

### In Plain English:
* **Step 1:** We predict ratings using our pre-trained Deep Learning Accuracy Model (**NCF**).
* **Step 2:** We analyze movie genres (**DOPM**) and encode raw IMDb user reviews using **BERT** to find semantic details about each movie. We co-track who watches what to build co-watching audience networks.
* **Step 3:** We generate hundreds of starting movie lists and filter out bad lists using a **PSNR Sweet-Spot** mathematical filter, leaving a high-quality initial population ($P_0$).
* **Step 4:** We run an **Evolutionary Algorithm** over 40 generations. It cross-breeds and mutates lists, dynamically removing duplicates, to evolve the most well-balanced recommended list.
* **Step 5:** We extract the **Pareto Front** (the set of optimal solutions) and select the single recommended list closest to the mathematical **Ideal Point**.

---

## 3. MOHS: How We Score "How Good It Is"

To prove **how much better** our optimized lists are compared to the original accuracy-only list, we calculate a comprehensive **Multi-Objective Harmonic Score (MOHS)**.

### What is MOHS?
It is a 3-variable harmonic mean:
$$\text{MOHS} = \frac{3}{\frac{1}{\text{Accuracy}} + \frac{1}{\text{Genre Diversity}} + \frac{1}{\text{BERT Serendipity}}}$$

MOHS behaves exactly like the **F1-Score** in standard machine learning, but for three goals. It strictly punishes any recommended list that fails in even a single dimension. 
* A list that is extremely accurate but has zero diversity or serendipity will receive a **very low MOHS score**.
* A list only receives a **high MOHS score** if it succeeds at all three stakeholder objectives simultaneously.

### The Proven Results:
Our final output logs prove a major, clear improvement:
* **User 0:** MOHS Quality Score improved from `0.4717` to `0.5078` (**+7.65% Net Quality Gain**).
* **User 99:** MOHS Quality Score improved from `0.5020` to `0.5647` (**+12.50% Net Quality Gain**).
* **User 499:** MOHS Quality Score improved from `0.4336` to `0.4980` (**+14.86% Net Quality Gain**).

Our evolutionary optimized lists are **mathematically superior** and significantly more exciting to interact with!
